"""
GetData V2 - 主入口
事件驱动的高频数据采集系统
"""
import asyncio
import sys

from playwright.async_api import async_playwright

from .clock import clock
from .event_bus import event_bus
from .aggregator import DataAggregator
from .writer import CSVWriter
from .sources.tencent import TencentSource
from .sources.sina import SinaSource
from .sources.baidu import BaiduSource, BAIDU_CONFIGS
from .sources.sentiment import SentimentSource


async def wait_for_data_ready(timeout: int = 60) -> bool:
    """等待各数据源就绪"""
    print(f"[Main] 等待数据源初始化 (超时: {timeout}秒)...")
    
    from .aggregator import DataAggregator
    
    # 创建临时聚合器检查缓存状态
    temp_agg = DataAggregator(on_row=lambda x: None)
    
    start_time = clock.now_s()
    
    while clock.now_s() - start_time < timeout:
        # 检查指数数据
        hsi_price = temp_agg.index_cache.get("HSI")
        hz_price = temp_agg.index_cache.get("HZ2083")
        index_ready = (hsi_price is not None and hsi_price > 0) or \
                      (hz_price is not None and hz_price > 0)
        
        # 检查期货数据
        fut_data = temp_agg.futures_cache.get("HSI")
        futures_ready = fut_data is not None
        
        # 检查汇率数据
        fx = temp_agg.fx_cache.get("HKDCNY")
        fx_ready = fx is not None and fx > 0
        
        if index_ready and fx_ready:
            elapsed = clock.now_s() - start_time
            print(f"[Main] 数据源就绪! (耗时 {elapsed:.1f}秒)")
            return True
        
        print(f"[Main] 预热中... Index:{index_ready} Fut:{futures_ready} FX:{fx_ready}")
        await asyncio.sleep(1)
    
    print("[Main] 警告: 预热超时，强行启动")
    return False


async def main():
    """主函数"""
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    print("=" * 60)
    print("  GetData V2 - 事件驱动数据采集系统")
    print("=" * 60)
    
    # 创建 Writer
    writer = CSVWriter(base_dir="./data")
    
    # 创建 Aggregator（写入回调）
    async def on_row(row):
        await writer.write(row)
    
    aggregator = DataAggregator(on_row=on_row)
    
    # 创建数据源
    tencent_source = TencentSource()
    sina_source = SinaSource()
    sentiment_source = SentimentSource()
    
    # 启动任务列表
    tasks = []
    
    # 启动 Aggregator
    task_aggregator = asyncio.create_task(aggregator.run())
    tasks.append(task_aggregator)
    
    # 启动 Sina 数据源
    task_sina = asyncio.create_task(sina_source.run())
    tasks.append(task_sina)
    
    # 启动 Sentiment 数据源
    task_sentiment = asyncio.create_task(sentiment_source.run())
    tasks.append(task_sentiment)
    
    # 启动 Playwright 和 Baidu 数据源
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        
        # 为每个百度配置创建数据源
        baidu_sources = [BaiduSource(cfg) for cfg in BAIDU_CONFIGS]
        for source in baidu_sources:
            task = asyncio.create_task(source.run_with_browser(browser))
            tasks.append(task)
        
        # 等待数据预热
        await asyncio.sleep(3)  # 给 Baidu WebSocket 一些时间连接
        
        # 启动主数据源 (Tencent)
        print("[Main] 启动 Tencent ETF 数据采集...")
        task_tencent = asyncio.create_task(tencent_source.run())
        tasks.append(task_tencent)
        
        print("[Main] 系统运行中... (Ctrl+C 停止)")
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("\n[Main] 收到停止信号...")
        except Exception as e:
            print(f"[Main] 错误: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序已停止")
