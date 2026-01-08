"""
DataAggregator - 数据聚合器
负责 CDC 去重、TTL 校验、乱序包处理、数据融合
"""
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, Awaitable
from collections import defaultdict

from .clock import clock
from .event_bus import event_bus, DataEvent, EventType


class TTLCache:
    """
    带 TTL 的缓存
    用于存储各数据源的最新值，自动过期
    """
    
    def __init__(self, ttl_ms: int = 5000):
        self.ttl_ms = ttl_ms
        self.data: Dict[str, Dict[str, Any]] = {}
    
    def set(self, key: str, value: Any, ts: float):
        """设置缓存值"""
        self.data[key] = {"value": value, "ts": ts}
    
    def get(self, key: str, current_ts: Optional[float] = None) -> Optional[Any]:
        """获取缓存值（带 TTL 检查）"""
        item = self.data.get(key)
        if not item:
            return None
        
        if current_ts is None:
            current_ts = clock.now_s()
        
        age_ms = (current_ts - item["ts"]) * 1000
        if age_ms > self.ttl_ms:
            return None  # 已过期
        
        return item["value"]
    
    def get_with_age(self, key: str) -> tuple:
        """获取缓存值和年龄（毫秒）"""
        item = self.data.get(key)
        if not item:
            return None, 99999
        
        age_ms = int((clock.now_s() - item["ts"]) * 1000)
        return item["value"], age_ms


class MonotonicVolumeTracker:
    """
    单调递增成交量跟踪器
    用于检测乱序包并计算增量成交量
    """
    
    def __init__(self):
        self.high_water_mark: Dict[str, Dict[str, float]] = {}
    
    def process(self, symbol: str, total_vol: float, total_amt: float) -> tuple:
        """
        处理成交量数据
        返回: (delta_vol, delta_amt, status)
        status: "ok" | "stale" | "first"
        """
        key = symbol
        
        if key not in self.high_water_mark:
            # 首次数据
            self.high_water_mark[key] = {"vol": total_vol, "amt": total_amt}
            return 0, 0, "first"
        
        hwm = self.high_water_mark[key]
        
        if total_vol < hwm["vol"]:
            # 乱序包：当前成交量小于已记录的最高值
            # 不更新 HWM，返回 stale 状态
            return 0, 0, "stale"
        
        # 正常增量
        delta_vol = total_vol - hwm["vol"]
        delta_amt = total_amt - hwm["amt"]
        
        # 更新 HWM
        self.high_water_mark[key] = {"vol": total_vol, "amt": total_amt}
        
        return delta_vol, delta_amt, "ok"


@dataclass
class AggregatedRow:
    """聚合后的数据行"""
    symbol: str
    tx_server_time: str
    tx_local_time: int
    
    # ETF 数据
    price: float = 0.0
    iopv: float = 0.0
    premium_rate: float = 0.0
    tick_vol: int = 0
    tick_amt: float = 0.0
    tick_vwap: float = 0.0
    
    # 盘口
    orderbook: Dict[str, Any] = field(default_factory=dict)
    
    # 关联数据
    index_price: float = 0.0
    fx_rate: float = 0.0
    sentiment: float = 0.0
    
    # 期货数据
    fut_price: float = 0.0
    fut_mid: float = 0.0
    fut_imb: float = 0.0
    fut_delta_vol: int = 0
    fut_pct: float = 0.0
    
    # 延迟信息
    idx_delay_ms: int = 99999
    fut_delay_ms: int = 99999
    
    # 数据标志
    data_flags: int = 0  # 0=正常, 1=首帧, 2=乱序
    
    def to_csv_row(self, has_futures: bool = False) -> list:
        """转换为 CSV 行"""
        row = [
            self.symbol,
            self.tx_server_time,
            self.tx_local_time,
            self.index_price if self.index_price > 0 else "",
            self.fx_rate,
            self.sentiment,
            self.price,
            self.iopv,
            round(self.premium_rate, 4),
            self.tick_vol,
            int(self.tick_amt),
            round(self.tick_vwap, 4),
        ]
        
        # 盘口数据
        for key in ["bp1", "bv1", "bp2", "bv2", "bp3", "bv3", "bp4", "bv4", "bp5", "bv5",
                    "sp1", "sv1", "sp2", "sv2", "sp3", "sv3", "sp4", "sv4", "sp5", "sv5"]:
            row.append(self.orderbook.get(key, 0))
        
        row.extend([self.idx_delay_ms, self.fut_delay_ms, self.data_flags])
        
        if has_futures:
            row.extend([
                self.fut_price,
                self.fut_mid,
                self.fut_imb,
                self.fut_delta_vol,
                self.fut_pct
            ])
        
        return row


class DataAggregator:
    """
    数据聚合器
    
    核心职责：
    1. CDC 去重：仅当 tx_server_time 变化时才输出
    2. TTL 校验：丢弃过期的关联数据
    3. 乱序包处理：使用 MonotonicVolumeTracker
    4. 数据融合：将 ETF 行情与指数、期货、情绪等数据融合
    """
    
    def __init__(self, on_row: Callable[[AggregatedRow], Awaitable[None]]):
        self.on_row = on_row  # 回调函数，用于输出聚合后的行
        
        # CDC：记录每个 symbol 最后一次输出的 server_ts
        self.last_output_ts: Dict[str, str] = {}
        
        # TTL 缓存
        self.index_cache = TTLCache(ttl_ms=5000)
        self.futures_cache = TTLCache(ttl_ms=3000)
        self.fx_cache = TTLCache(ttl_ms=10000)
        self.sentiment_cache = TTLCache(ttl_ms=5000)
        
        # 成交量跟踪器
        self.volume_tracker = MonotonicVolumeTracker()
        
        # ETF 配置
        self.etf_config = {
            "sz159920": {"index_key": "HSI", "future_key": "HSI", "sentiment_key": "HSI", "has_futures": True},
            "sh513130": {"index_key": "HZ2083", "future_key": None, "sentiment_key": "HSTECH", "has_futures": False}
        }
    
    async def run(self):
        """主循环：从 EventBus 消费事件"""
        print("[Aggregator] 启动...")
        
        while True:
            event = await event_bus.get(timeout=1.0)
            if event:
                await self._process_event(event)
    
    async def _process_event(self, event: DataEvent):
        """处理单个事件"""
        
        if event.event_type == EventType.INDEX_PRICE:
            # 更新指数缓存
            self.index_cache.set(
                event.symbol,
                event.payload.get("price", 0),
                event.recv_time
            )
        
        elif event.event_type == EventType.FUTURES_QUOTE:
            # 更新期货缓存
            self.futures_cache.set(
                event.symbol,
                event.payload,
                event.recv_time
            )
        
        elif event.event_type == EventType.FX_RATE:
            # 更新汇率缓存
            self.fx_cache.set(
                event.symbol,
                event.payload.get("price", 0),
                event.recv_time
            )
        
        elif event.event_type == EventType.SENTIMENT:
            # 更新情绪缓存
            self.sentiment_cache.set(
                event.symbol,
                event.payload.get("score", 0),
                event.recv_time
            )
        
        elif event.event_type == EventType.ETF_QUOTE:
            # 核心逻辑：处理 ETF 行情
            await self._process_etf_quote(event)
    
    async def _process_etf_quote(self, event: DataEvent):
        """处理 ETF 行情事件"""
        symbol = event.symbol
        server_ts = event.server_ts
        
        # CDC 去重：检查 server_ts 是否变化
        if symbol in self.last_output_ts:
            if server_ts == self.last_output_ts[symbol]:
                # 重复数据，跳过
                return
        
        # 更新 CDC 记录
        self.last_output_ts[symbol] = server_ts
        
        # 获取配置
        config = self.etf_config.get(symbol, {})
        index_key = config.get("index_key", "")
        future_key = config.get("future_key", "")
        sentiment_key = config.get("sentiment_key", "")
        has_futures = config.get("has_futures", False)
        
        # 获取关联数据（带 TTL 检查）
        current_ts = clock.now_s()
        
        index_price, idx_delay = self.index_cache.get_with_age(index_key)
        if index_price is None:
            index_price = 0.0
            idx_delay = 99999
        
        fx_rate = self.fx_cache.get("HKDCNY", current_ts) or 0.0
        sentiment = self.sentiment_cache.get(sentiment_key, current_ts) or 0.0
        
        # 期货数据
        fut_data = {}
        fut_delay = 99999
        if future_key:
            fut_data, fut_delay = self.futures_cache.get_with_age(future_key)
            if fut_data is None:
                fut_data = {}
                fut_delay = 99999
        
        # 处理成交量
        payload = event.payload
        total_vol = payload.get("total_vol", 0)
        total_amt = payload.get("total_amt", 0)
        
        delta_vol, delta_amt, status = self.volume_tracker.process(symbol, total_vol, total_amt)
        
        # 确定 data_flags
        if status == "first":
            data_flags = 1
        elif status == "stale":
            data_flags = 2
            delta_vol = 0
            delta_amt = 0
        else:
            data_flags = 0
        
        # 计算 VWAP
        tick_vwap = delta_amt / delta_vol if delta_vol > 0 else payload.get("price", 0)
        
        # 构建聚合行
        row = AggregatedRow(
            symbol=symbol,
            tx_server_time=server_ts,
            tx_local_time=event.local_ts,
            price=payload.get("price", 0),
            iopv=payload.get("iopv", 0),
            premium_rate=payload.get("premium_rate", 0),
            tick_vol=int(delta_vol),
            tick_amt=delta_amt,
            tick_vwap=tick_vwap,
            orderbook=payload.get("orderbook", {}),
            index_price=index_price,
            fx_rate=fx_rate,
            sentiment=sentiment,
            fut_price=fut_data.get("price", 0) if fut_data else 0,
            fut_mid=fut_data.get("mid", 0) if fut_data else 0,
            fut_imb=fut_data.get("imb", 0) if fut_data else 0,
            fut_delta_vol=fut_data.get("delta_vol", 0) if fut_data else 0,
            fut_pct=fut_data.get("pct", 0) if fut_data else 0,
            idx_delay_ms=idx_delay,
            fut_delay_ms=fut_delay,
            data_flags=data_flags
        )
        
        # 输出
        await self.on_row(row)
        
        # 打印状态
        print(f"[{server_ts}] {symbol} | P:{payload.get('price')} | Vol:{int(delta_vol)} | Idx:{index_price:.0f} | Flag:{data_flags}")
