"""
BaiduSource - 百度股市通指数数据源
使用 Playwright 通过 WebSocket 获取实时指数价格
"""
import asyncio
import json
from datetime import datetime
from typing import Optional, Dict

from playwright.async_api import async_playwright, Browser

from ..clock import clock
from ..event_bus import event_bus, DataEvent, EventType


# 百度配置
BAIDU_CONFIGS = [
    {"url": "https://gushitong.baidu.com/index/hk-HSI", "key": "HSI"},
    {"url": "https://gushitong.baidu.com/index/hk-HZ2083", "key": "HZ2083"}
]


class BaiduSource:
    """
    百度股市通指数数据源
    
    通过 Playwright 打开页面，监听 WebSocket 消息
    获取恒生指数 (HSI) 和恒生科技指数 (HZ2083) 的实时价格
    """
    
    def __init__(self, config: Dict):
        self.url = config["url"]
        self.key = config["key"]
    
    async def handle_websocket_message(self, msg: str):
        """处理 WebSocket 消息"""
        try:
            data = json.loads(msg)
            if "data" not in data:
                return
            
            raw_data = data["data"]
            if "cur" not in raw_data:
                return
            
            cur = raw_data["cur"]
            price = float(cur.get("price", 0))
            
            if price <= 0:
                return
            
            # 解析服务端时间戳
            server_ts = "N/A"
            if "point" in raw_data and "realTimeStampMs" in raw_data["point"]:
                try:
                    ms = int(raw_data["point"]["realTimeStampMs"])
                    dt = datetime.fromtimestamp(ms / 1000.0)
                    server_ts = dt.strftime("%H:%M:%S")
                except:
                    pass
            
            recv_time = clock.now_s()
            
            event = DataEvent(
                event_type=EventType.INDEX_PRICE,
                source="baidu",
                symbol=self.key,
                local_ts=clock.now_ms(),
                server_ts=server_ts,
                recv_time=recv_time,
                tick_id=clock.next_tick(),
                payload={"price": price}
            )
            
            await event_bus.publish(event)
            
        except Exception:
            pass
    
    async def run_with_browser(self, browser: Browser):
        """使用已有的浏览器实例运行"""
        context = await browser.new_context()
        page = await context.new_page()
        
        def on_websocket(ws):
            async def handle_msg(payload):
                content = payload.decode('utf-8') if isinstance(payload, bytes) else payload
                await self.handle_websocket_message(content)
            ws.on("framereceived", lambda payload: asyncio.create_task(handle_msg(payload)))
        
        page.on("websocket", on_websocket)
        
        try:
            print(f"[BaiduSource:{self.key}] 启动...")
            await page.goto(self.url, wait_until="domcontentloaded", timeout=60000)
            await asyncio.Future()  # 永久等待
        except Exception as e:
            print(f"[BaiduSource:{self.key}] 错误: {e}")
        finally:
            await context.close()


async def create_baidu_sources() -> list:
    """创建所有百度数据源实例"""
    return [BaiduSource(cfg) for cfg in BAIDU_CONFIGS]
