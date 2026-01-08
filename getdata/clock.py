"""
GlobalClock - 统一时钟模块
提供全局唯一的 Tick ID 和毫秒级时间戳
"""
import time
from dataclasses import dataclass
from typing import Optional


class GlobalClock:
    """
    全局时钟单例，用于：
    1. 生成单调递增的 tick_id
    2. 提供统一的毫秒时间戳
    3. 计算数据延迟
    """
    _instance: Optional['GlobalClock'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tick_id = 0
            cls._instance._start_time = time.time()
        return cls._instance
    
    @property
    def tick_id(self) -> int:
        return self._tick_id
    
    def next_tick(self) -> int:
        """生成下一个 Tick ID（单调递增）"""
        self._tick_id += 1
        return self._tick_id
    
    def now_ms(self) -> int:
        """当前毫秒时间戳"""
        return int(time.time() * 1000)
    
    def now_s(self) -> float:
        """当前秒级时间戳（浮点）"""
        return time.time()
    
    def elapsed_since_start(self) -> float:
        """自启动以来经过的秒数"""
        return time.time() - self._start_time
    
    def calculate_delay_ms(self, recv_time: float) -> int:
        """计算数据延迟（毫秒）"""
        if recv_time <= 0:
            return 99999
        return int((time.time() - recv_time) * 1000)


# 全局单例
clock = GlobalClock()
