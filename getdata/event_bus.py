"""
EventBus - 事件总线模块
所有数据源通过事件总线解耦通信
"""
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum


class EventType(Enum):
    """事件类型枚举"""
    ETF_QUOTE = "etf_quote"           # ETF 行情
    INDEX_PRICE = "index_price"       # 指数价格
    FUTURES_QUOTE = "futures_quote"   # 期货行情
    FX_RATE = "fx_rate"               # 汇率
    SENTIMENT = "sentiment"           # 情绪因子


@dataclass
class DataEvent:
    """
    统一数据事件结构
    所有 Worker 产生的数据都封装为此格式
    """
    event_type: EventType
    source: str              # "tencent" | "sina" | "baidu" | "sentiment"
    symbol: str              # "sz159920" | "HSI" | "HZ2083" | ...
    local_ts: int            # 本地毫秒时间戳
    server_ts: str           # 服务端时间字符串（如有）
    recv_time: float         # 接收时间（秒级浮点，用于 TTL 计算）
    tick_id: int             # 全局 Tick ID
    payload: Dict[str, Any] = field(default_factory=dict)  # 原始数据
    
    def __repr__(self):
        return f"<Event {self.event_type.value}:{self.symbol} @{self.server_ts}>"


class EventBus:
    """
    异步事件总线
    - 生产者：各 Worker 通过 publish() 发送事件
    - 消费者：Aggregator 通过 subscribe() 接收事件
    """
    def __init__(self, maxsize: int = 1000):
        self._queue: asyncio.Queue[DataEvent] = asyncio.Queue(maxsize=maxsize)
        self._subscribers: list = []
    
    async def publish(self, event: DataEvent) -> None:
        """发布事件到总线"""
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # 队列满时丢弃最旧的事件
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
            except:
                pass
    
    async def get(self, timeout: float = 1.0) -> Optional[DataEvent]:
        """获取下一个事件（带超时）"""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    def qsize(self) -> int:
        """当前队列大小"""
        return self._queue.qsize()


# 全局事件总线单例
event_bus = EventBus()
