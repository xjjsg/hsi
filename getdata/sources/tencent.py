"""
TencentSource - 腾讯 ETF 行情数据源
从 qt.gtimg.cn 获取 ETF 实时行情
"""
import asyncio
import aiohttp
from datetime import datetime, time as dt_time
from typing import Optional, Dict, Any

from ..clock import clock
from ..event_bus import event_bus, DataEvent, EventType


# ETF 配置
ETF_CONFIG = {
    "sz159920": {
        "index_key": "HSI",
        "future_key": "HSI",
        "sentiment_key": "HSI",
        "has_futures": True
    },
    "sh513130": {
        "index_key": "HZ2083",
        "future_key": None,
        "sentiment_key": "HSTECH",
        "has_futures": False
    }
}


class TencentSource:
    """
    腾讯 ETF 行情数据源
    
    改进点：
    1. 不再直接写文件，而是发送事件到 EventBus
    2. 使用 GlobalClock 统一时间戳
    3. 保留原始数据，由 Aggregator 做去重
    """
    
    def __init__(self):
        self.last_snapshot: Dict[str, Dict[str, Any]] = {}
        
        # 交易时间段
        self.T1_START = dt_time(9, 29, 50)
        self.T1_END = dt_time(11, 30, 10)
        self.T2_START = dt_time(12, 0, 0)   # 午间快照开始
        self.T2_END = dt_time(12, 0, 30)    # 午间快照结束 (30秒窗口)
        self.T3_START = dt_time(12, 59, 50)
        self.T3_END = dt_time(15, 0, 10)
    
    def is_trading_time(self) -> bool:
        """检查是否在交易时间内"""
        now = datetime.now().time()
        if self.T1_START <= now <= self.T1_END:
            return True
        if self.T2_START <= now <= self.T2_END:
            return True  # 午间快照
        if self.T3_START <= now <= self.T3_END:
            return True
        return False
    
    def parse_line(self, line: str) -> Optional[DataEvent]:
        """解析腾讯行情数据"""
        try:
            if '="' not in line:
                return None
            
            var_part, content = line.split('="')
            symbol = var_part.split('_')[-1]
            f = content.strip('";\\n').split('~')
            
            if len(f) < 80:
                return None
            
            # 解析服务端时间
            t_str = f[30]
            server_ts = f"{t_str[8:10]}:{t_str[10:12]}:{t_str[12:14]}"
            
            # 获取价格和成交量
            price = float(f[3])
            total_vol = float(f[6]) * 100
            
            try:
                total_amt = float(f[35].split('/')[2])
            except:
                total_amt = float(f[37]) * 10000
            
            # IOPV
            iopv = float(f[78]) if f[78] else 0.0
            premium_rate = (price - iopv) / iopv * 100 if iopv > 0 else 0.0
            
            # 盘口数据
            def safe_int_vol(idx):
                return int(f[idx]) * 100
            
            orderbook = {
                "bp1": f[9], "bv1": safe_int_vol(10),
                "bp2": f[11], "bv2": safe_int_vol(12),
                "bp3": f[13], "bv3": safe_int_vol(14),
                "bp4": f[15], "bv4": safe_int_vol(16),
                "bp5": f[17], "bv5": safe_int_vol(18),
                "sp1": f[19], "sv1": safe_int_vol(20),
                "sp2": f[21], "sv2": safe_int_vol(22),
                "sp3": f[23], "sv3": safe_int_vol(24),
                "sp4": f[25], "sv4": safe_int_vol(26),
                "sp5": f[27], "sv5": safe_int_vol(28),
            }
            
            recv_time = clock.now_s()
            
            # 构建事件
            event = DataEvent(
                event_type=EventType.ETF_QUOTE,
                source="tencent",
                symbol=symbol,
                local_ts=clock.now_ms(),
                server_ts=server_ts,
                recv_time=recv_time,
                tick_id=clock.next_tick(),
                payload={
                    "price": price,
                    "iopv": iopv,
                    "premium_rate": round(premium_rate, 4),
                    "total_vol": int(total_vol),
                    "total_amt": total_amt,
                    "orderbook": orderbook,
                    "config": ETF_CONFIG.get(symbol, {}),
                    "raw_fields": f  # 保留原始字段用于调试
                }
            )
            
            return event
            
        except Exception as e:
            return None
    
    async def run(self):
        """主循环：每秒抓取一次数据"""
        async with aiohttp.ClientSession() as session:
            codes_str = ",".join(ETF_CONFIG.keys())
            url = f"http://qt.gtimg.cn/q={codes_str}"
            
            print("[TencentSource] 启动...")
            
            while True:
                start_ts = clock.now_s()
                
                try:
                    if self.is_trading_time():
                        async with session.get(url) as resp:
                            text = await resp.text(encoding='gbk')
                        
                        lines = text.strip().split(';')
                        for line in lines:
                            if not line.strip():
                                continue
                            event = self.parse_line(line)
                            if event:
                                await event_bus.publish(event)
                    
                except Exception as e:
                    pass
                
                # 每秒轮询一次
                elapsed = clock.now_s() - start_ts
                await asyncio.sleep(max(0, 1.0 - elapsed))
