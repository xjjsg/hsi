"""
SentimentSource - 情绪因子数据源
基于港股成分股的实时成交数据计算加权情绪分数
"""
import asyncio
import math
import aiohttp
import numpy as np
from typing import Dict, Optional

from ..clock import clock
from ..event_bus import event_bus, DataEvent, EventType


# 成分股权重
HSI_WEIGHTS = {
    '00005': 8.0, '00700': 8.0, '09988': 8.0, '01299': 7.0, '00939': 5.5,
    '03690': 5.5, '00941': 4.0, '01398': 3.5, '00388': 3.0, '01211': 2.5
}

HSTECH_WEIGHTS = {
    '01810': 15.0, '03690': 13.0, '00700': 10.0, '09988': 8.0, '09618': 6.0,
    '01024': 6.0,  '02015': 5.0,  '09961': 4.0,  '00999': 3.5, '00981': 3.0
}


class SingleStockCalculator:
    """单只股票情绪计算器"""
    
    def __init__(self, code: str):
        self.code = code
        self.last_vol = 0
        self.last_amt = 0
        self.vol_history = []
        self.score_ema = 0
        self.initialized = False
    
    def calculate(self, data: Dict) -> float:
        curr_vol = data['vol']
        curr_amt = data['amount']
        curr_price = data['price']
        
        if not self.initialized:
            self.last_vol = curr_vol
            self.last_amt = curr_amt
            self.initialized = True
            return 0
        
        delta_vol = curr_vol - self.last_vol
        delta_amt = curr_amt - self.last_amt
        
        # 防止负成交量（重置保护）
        if delta_vol < 0:
            self.last_vol = curr_vol
            self.last_amt = curr_amt
            return self.score_ema
        
        self.last_vol = curr_vol
        self.last_amt = curr_amt
        
        if delta_vol == 0:
            self.score_ema *= 0.98
            return self.score_ema
        
        vwap_3s = delta_amt / delta_vol
        diff_bp = (curr_price - vwap_3s) / vwap_3s * 10000
        score_momentum = float(np.clip(diff_bp * 1.5, -4, 4))
        
        score_structure = 0
        bid1 = data.get('bid1', 0)
        ask1 = data.get('ask1', 0)
        
        if bid1 > 0 and ask1 > 0:
            if curr_price >= ask1:
                score_structure = 2
            elif curr_price <= bid1:
                score_structure = -2
        else:
            score_momentum *= 1.2
        
        context_factor = 1.0
        day_low = data.get('low', curr_price)
        day_high = data.get('high', curr_price)
        
        if abs(curr_price - day_low) / day_low < 0.005 and score_momentum < 0:
            context_factor = 1.5
        elif abs(curr_price - day_high) / day_high < 0.005 and score_momentum > 0:
            context_factor = 1.2
        
        self.vol_history.append(delta_vol)
        if len(self.vol_history) > 20:
            self.vol_history.pop(0)
        avg_vol = np.mean(self.vol_history) + 100
        vol_ratio = min(delta_vol / avg_vol, 4.0)
        
        raw_score = (score_momentum + score_structure) * vol_ratio * context_factor
        self.score_ema = self.score_ema * 0.7 + raw_score * 0.3
        
        return self.score_ema


class SentimentSource:
    """
    情绪因子数据源
    
    输出事件类型：SENTIMENT
    """
    
    def __init__(self):
        all_codes = set(list(HSI_WEIGHTS.keys()) + list(HSTECH_WEIGHTS.keys()))
        self.calculators = {code: SingleStockCalculator(code) for code in all_codes}
        self.hsi_norm_weights = self._normalize_weights(HSI_WEIGHTS)
        self.hstech_norm_weights = self._normalize_weights(HSTECH_WEIGHTS)
    
    def _normalize_weights(self, w_dict: Dict) -> Dict:
        total = sum(w_dict.values())
        return {k: v/total for k, v in w_dict.items()}
    
    def parse_stock_line(self, line: str) -> Optional[Dict]:
        try:
            if '="' not in line:
                return None
            _, content = line.split('="')
            parts = content.strip('";\\n').split('~')
            if len(parts) < 38:
                return None
            return {
                'code': parts[2],
                'price': float(parts[3]),
                'vol': float(parts[6]),
                'bid1': float(parts[9]),
                'ask1': float(parts[19]),
                'time': parts[31],
                'high': float(parts[34]),
                'low': float(parts[35]),
                'amount': float(parts[37]) if len(parts) > 37 and float(parts[37]) > 10000 else float(parts[38])
            }
        except Exception:
            return None
    
    async def run(self):
        """主循环：每秒计算一次情绪分数"""
        async with aiohttp.ClientSession() as session:
            codes_str = ",".join([f"r_hk{c}" for c in self.calculators.keys()])
            url = f"http://qt.gtimg.cn/q={codes_str}"
            
            print("[SentimentSource] 启动...")
            
            while True:
                start_ts = clock.now_s()
                
                try:
                    async with session.get(url) as resp:
                        text = await resp.text()
                    
                    lines = text.strip().split(';')
                    stock_scores = {}
                    
                    for line in lines:
                        if len(line) < 10:
                            continue
                        data = self.parse_stock_line(line)
                        if data and data['code'] in self.calculators:
                            score = self.calculators[data['code']].calculate(data)
                            stock_scores[data['code']] = score
                    
                    # 计算加权分数
                    hsi_final = sum(
                        stock_scores.get(code, 0) * weight
                        for code, weight in self.hsi_norm_weights.items()
                    )
                    hstech_final = sum(
                        stock_scores.get(code, 0) * weight
                        for code, weight in self.hstech_norm_weights.items()
                    )
                    
                    recv_time = clock.now_s()
                    
                    # 发布 HSI 情绪事件
                    await event_bus.publish(DataEvent(
                        event_type=EventType.SENTIMENT,
                        source="sentiment",
                        symbol="HSI",
                        local_ts=clock.now_ms(),
                        server_ts="N/A",
                        recv_time=recv_time,
                        tick_id=clock.next_tick(),
                        payload={"score": round(math.tanh(hsi_final * 0.25) * 10, 4)}
                    ))
                    
                    # 发布 HSTECH 情绪事件
                    await event_bus.publish(DataEvent(
                        event_type=EventType.SENTIMENT,
                        source="sentiment",
                        symbol="HSTECH",
                        local_ts=clock.now_ms(),
                        server_ts="N/A",
                        recv_time=recv_time,
                        tick_id=clock.next_tick(),
                        payload={"score": round(math.tanh(hstech_final * 0.25) * 10, 4)}
                    ))
                    
                except Exception:
                    pass
                
                elapsed = clock.now_s() - start_ts
                await asyncio.sleep(max(0, 1.0 - elapsed))
