"""
SinaSource - 新浪期货和汇率数据源
从 hq.sinajs.cn 获取恒生指数期货和港币汇率
"""
import asyncio
import aiohttp
from datetime import datetime
from typing import Optional

from ..clock import clock
from ..event_bus import event_bus, DataEvent, EventType


# 配置
SINA_FUTURES_CONFIG = ["hf_HSI"]
SINA_FX_CONFIG = "HKDCNY"


class SinaSource:
    """
    新浪期货和汇率数据源
    
    输出事件类型：
    - FUTURES_QUOTE: 期货行情
    - FX_RATE: 汇率
    """
    
    def __init__(self):
        self.last_cum_vols = {}
        self.headers = {
            "Referer": "https://finance.sina.com.cn/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"
        }
    
    def safe_float(self, val) -> float:
        try:
            if not val or not str(val).strip():
                return 0.0
            return float(val)
        except (ValueError, TypeError):
            return 0.0
    
    def parse_futures(self, line: str, code: str) -> Optional[DataEvent]:
        """解析期货数据"""
        try:
            split_idx = line.find('="')
            if split_idx == -1:
                return None
            
            data_str = line[split_idx+2:].strip('";')
            if not data_str:
                return None
            
            f = data_str.split(',')
            if len(f) < 14:
                return None
            
            price = self.safe_float(f[0])
            bid1 = self.safe_float(f[2])
            ask1 = self.safe_float(f[3])
            tick_time = f[6]
            prev_close = self.safe_float(f[7])
            bid_vol = self.safe_float(f[10])
            ask_vol = self.safe_float(f[11])
            cum_vol = self.safe_float(f[14])
            
            # 计算中间价、不平衡度、增量成交量
            mid = (bid1 + ask1) / 2 if (bid1 > 0 and ask1 > 0) else price
            total_depth = bid_vol + ask_vol
            imb = (bid_vol - ask_vol) / total_depth if total_depth > 0 else 0.0
            
            last_vol = self.last_cum_vols.get(code, cum_vol)
            delta_vol = cum_vol - last_vol if cum_vol >= last_vol else 0
            self.last_cum_vols[code] = cum_vol
            
            pct = (price - prev_close) / prev_close * 100 if prev_close > 0 else 0.0
            
            recv_time = clock.now_s()
            
            return DataEvent(
                event_type=EventType.FUTURES_QUOTE,
                source="sina",
                symbol=code,
                local_ts=clock.now_ms(),
                server_ts=tick_time,
                recv_time=recv_time,
                tick_id=clock.next_tick(),
                payload={
                    "price": price,
                    "mid": round(mid, 2),
                    "imb": round(imb, 2),
                    "delta_vol": int(delta_vol),
                    "pct": round(pct, 2),
                    "bid1": bid1,
                    "ask1": ask1
                }
            )
            
        except Exception:
            return None
    
    def parse_fx(self, line: str) -> Optional[DataEvent]:
        """解析汇率数据"""
        try:
            split_idx = line.find('="')
            if split_idx == -1:
                return None
            
            data_str = line[split_idx+2:].strip('";')
            if not data_str:
                return None
            
            f = data_str.split(',')
            if len(f) < 4:
                return None
            
            price = self.safe_float(f[3])
            server_time = f[0] if f[0] else "N/A"
            
            if price <= 0:
                return None
            
            recv_time = clock.now_s()
            
            return DataEvent(
                event_type=EventType.FX_RATE,
                source="sina",
                symbol="HKDCNY",
                local_ts=clock.now_ms(),
                server_ts=server_time,
                recv_time=recv_time,
                tick_id=clock.next_tick(),
                payload={"price": price}
            )
            
        except Exception:
            return None
    
    async def run(self):
        """主循环：每 0.5 秒抓取一次"""
        codes = SINA_FUTURES_CONFIG + [SINA_FX_CONFIG]
        url = f"http://hq.sinajs.cn/list={','.join(codes)}"
        
        async with aiohttp.ClientSession() as session:
            print("[SinaSource] 启动...")
            
            while True:
                start_ts = clock.now_s()
                
                try:
                    async with session.get(url, headers=self.headers) as resp:
                        text = await resp.text(encoding='gbk')
                    
                    lines = text.strip().split('\n')
                    for line in lines:
                        if '="' not in line:
                            continue
                        
                        if "hf_" in line:
                            # 期货数据
                            code = line.split('_')[-1].split('=')[0]
                            event = self.parse_futures(line, code)
                            if event:
                                await event_bus.publish(event)
                        
                        elif "HKDCNY" in line:
                            # 汇率数据
                            event = self.parse_fx(line)
                            if event:
                                await event_bus.publish(event)
                
                except Exception:
                    pass
                
                elapsed = clock.now_s() - start_ts
                await asyncio.sleep(max(0, 0.5 - elapsed))
