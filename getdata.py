import asyncio
import aiohttp
import aiofiles
import csv
import os
import time
import sys
import json
from datetime import datetime
from typing import Dict, List, NamedTuple
from playwright.async_api import async_playwright, Browser

# ==============================================================================
#                                MODULE A: SINA (新浪接口)
# ==============================================================================

# --- A1. 配置 ---
SINA_SYMBOL_FILE_MAP = {
    "sz159920": "sz159920data.csv",  # 恒生ETF
    "sh513130": "sh513130.csv"       # 恒生科技
}

SINA_WS_URL = f"wss://w.sinajs.cn/wskt?list={','.join(SINA_SYMBOL_FILE_MAP.keys())}"

SINA_HEADERS = {
    "Origin": "https://quotes.sina.cn",
    "Host": "w.sinajs.cn",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
}

# --- A2. 数据结构 ---
SINA_LAST_SNAPSHOT: Dict[str, Dict[str, float]] = {}

# CSV 表头
SINA_CSV_HEADER = [
    "local_ts", "symbol", "ex_time", "price", "interval_s",
    "tick_vol", "tick_amt", "tick_vwap",
    "bp1", "bv1", "bp2", "bv2", "bp3", "bv3", "bp4", "bv4", "bp5", "bv5",
    "sp1", "sv1", "sp2", "sv2", "sp3", "sv3", "sp4", "sv4", "sp5", "sv5"
]

# --- A3. 核心逻辑 ---
async def sina_parse_and_process(msg: str, queue: asyncio.Queue):
    global SINA_LAST_SNAPSHOT
    
    try:
        current_ts = time.time()
        local_ts_ms = int(current_ts * 1000)
        
        lines = msg.strip().split('\n')
        for line in lines:
            if not line: continue
            
            try:
                left, right = line.split('=', 1)
                symbol = left.split('_')[-1]
                
                if symbol not in SINA_SYMBOL_FILE_MAP: continue
                
                d = right.split(',')
                if len(d) < 32: continue
                
                current_price = float(d[3])
                cur_cum_vol = float(d[8])
                cur_cum_amt = float(d[9])
                ex_time = d[31]
                
                # --- 增量计算 ---
                if symbol in SINA_LAST_SNAPSHOT:
                    prev = SINA_LAST_SNAPSHOT[symbol]
                    tick_vol = cur_cum_vol - prev['vol']
                    tick_amt = cur_cum_amt - prev['amt']
                    interval = current_ts - prev['ts']
                    
                    if tick_vol > 0:
                        tick_vwap = tick_amt / tick_vol
                    else:
                        tick_vwap = current_price
                else:
                    tick_vol = 0.0
                    tick_amt = 0.0
                    tick_vwap = current_price
                    interval = 0.0
                
                SINA_LAST_SNAPSHOT[symbol] = {
                    'vol': cur_cum_vol, 'amt': cur_cum_amt, 'ts': current_ts
                }
                
                # --- 组装数据 ---
                row = [
                    local_ts_ms, symbol, ex_time, current_price, round(interval, 3),
                    int(tick_vol), int(tick_amt), round(tick_vwap, 4),
                    # 买1-5
                    d[11], d[10], d[13], d[12], d[15], d[14], d[17], d[16], d[19], d[18],
                    # 卖1-5
                    d[21], d[20], d[23], d[22], d[25], d[24], d[27], d[26], d[29], d[28]
                ]            
                queue.put_nowait(row)
                print(f"\r[Sina] {ex_time} {symbol} P:{current_price} Vol:{int(tick_vol)}   ", end="")
                
            except Exception:
                continue
    except Exception as e:
        print(f"[Sina] 解析错误: {e}")

async def sina_flush_buffer(filename: str, buffer: list):
    """辅助函数：将指定缓冲区写入文件"""
    try:
        async with aiofiles.open(filename, "a", newline="", encoding="utf-8") as f:
            lines = [",".join(map(str, row)) + "\n" for row in buffer]
            await f.writelines(lines)
        buffer.clear()
    except Exception as e:
        print(f"[Sina] 写入 {filename} 失败: {e}")

async def sina_csv_writer(queue: asyncio.Queue):
    # 1. 初始化所有文件
    for symbol, fname in SINA_SYMBOL_FILE_MAP.items():
        if not os.path.exists(fname):
            print(f"[Sina] 初始化文件: {fname}")
            with open(fname, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(SINA_CSV_HEADER)
    
    # 2. 初始化独立缓冲区
    buffers = {fname: [] for fname in SINA_SYMBOL_FILE_MAP.values()}
    
    while True:
        item = await queue.get()
        symbol = item[1]
        target_file = SINA_SYMBOL_FILE_MAP.get(symbol)
        
        if target_file:
            buffers[target_file].append(item)
        
        queue.task_done()
        
        # 写入策略: 满20条刷入
        if target_file and len(buffers[target_file]) >= 20:
            await sina_flush_buffer(target_file, buffers[target_file])
            
        # 队列空闲时，刷入剩余数据
        if queue.empty():
            for fname, buf in buffers.items():
                if buf:
                    await sina_flush_buffer(fname, buf)

async def run_sina_task(queue: asyncio.Queue):
    """Sina 的 WebSocket 监听循环"""
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(SINA_WS_URL, headers=SINA_HEADERS) as ws:
                    print("\n[Sina] WebSocket 连接成功!")
                    
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await sina_parse_and_process(msg.data, queue)
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            print("\n[Sina] 连接断开")
                            break
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            print("\n[Sina] 连接错误")
                            break
        except Exception as e:
            print(f"\n[Sina] 网络波动: {e} | 3秒后重连...")
            await asyncio.sleep(3)

# ==============================================================================
#                                MODULE B: BAIDU (百度股市通)
# ==============================================================================

# --- B1. 配置 ---
class WsScrapeConfigBaidu(NamedTuple):
    page_url: str      
    code: str          
    csv_file: str      

BAIDU_HEADER = ["data_time", "price"]

# --- B2. 辅助函数 ---
def baidu_initialize_csv(csv_file: str, header: list):
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)
        print(f"[Baidu] 📄 新建文件: {csv_file}")

async def baidu_async_write_csv(csv_file: str, row: list):
    """异步写入 CSV (Baidu模式)"""
    try:
        line = ",".join(map(str, row)) + "\n"
        async with aiofiles.open(csv_file, "a", encoding="utf-8") as f:
            await f.write(line)
    except Exception as e:
        print(f"[Baidu] ❌ 写入失败: {e}")

# --- B3. 核心解析 ---
async def baidu_parse_message(msg: str, csv_file: str, target_code: str):
    try:
        if not msg: return
        content = msg.decode('utf-8') if isinstance(msg, bytes) else msg
        
        try:
            json_data = json.loads(content)
        except json.JSONDecodeError:
            return 

        raw_data = json_data.get("data")

        # 过滤掉 "pong" 和非字典数据
        if isinstance(raw_data, str) or not isinstance(raw_data, dict):
            return 

        # 确认股票代码
        msg_code = raw_data.get("code", "")
        if target_code not in msg_code and msg_code != "":
            return

        # 提取数据
        cur_data = raw_data.get("cur")
        if not cur_data:
            return

        # 获取价格
        price = cur_data.get("price") or cur_data.get("close") or cur_data.get("avgPrice")
        if not price: return

        # 获取时间
        timestamp = int(datetime.now().timestamp() * 1000)
        data_time = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        row = [data_time, price]
        
        await baidu_async_write_csv(csv_file, row)
        # 为了不干扰Sina的print，这里稍微改一下格式
        # print(f"✅ [Baidu-{target_code}] {price}") 

    except Exception as e:
        pass 

# --- B4. 运行逻辑 ---
async def run_baidu_page(config: WsScrapeConfigBaidu, browser: Browser):
    print(f"[Baidu] 启动: {config.code}")
    context = await browser.new_context()
    page = await context.new_page()

    def on_web_socket(ws):
        ws.on("framereceived", lambda payload: asyncio.create_task(
            baidu_parse_message(payload, config.csv_file, config.code)
        ))

    page.on("websocket", on_web_socket)

    try:
        await page.goto(config.page_url, wait_until="domcontentloaded", timeout=60000)
        # 永久等待，直到被取消
        await asyncio.Future()
    except asyncio.CancelledError:
        print(f"[Baidu] 任务停止: {config.code}")
    except Exception as e:
        print(f"[Baidu] ❌ [{config.code}] 中断: {e}")
    finally:
        await context.close()

# ==============================================================================
#                                MAIN ENTRY POINT
# ==============================================================================

async def main():
    # 1. Windows 系统适配
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # ----------------- 初始化 Sina 模块 -----------------
    sina_queue = asyncio.Queue()
    
    # 启动 Sina 写入器 (后台任务)
    sina_writer_task = asyncio.create_task(sina_csv_writer(sina_queue))
    
    # ----------------- 初始化 Baidu 模块 -----------------
    baidu_configs = [
        WsScrapeConfigBaidu("https://gushitong.baidu.com/index/hk-HSI", "HSI", "HSI.csv"),
        WsScrapeConfigBaidu("https://gushitong.baidu.com/index/hk-HZ2083", "HZ2083", "HTEC_HZ2083.csv")
    ]

    for c in baidu_configs:
        baidu_initialize_csv(c.csv_file, BAIDU_HEADER)

    # ----------------- 启动所有任务 -----------------
    print(">>> 系统启动中: 正在整合 Sina 与 Baidu 数据源...")

    async with async_playwright() as p:
        # 启动浏览器 (headless=True 为后台运行)
        browser = await p.chromium.launch(headless=True)
        
        # 创建任务列表
        tasks = []
        
        # 1. 添加 Sina 监听任务
        tasks.append(asyncio.create_task(run_sina_task(sina_queue)))
        
        # 2. 添加 Baidu 监听任务 (每个配置一个页面)
        for config in baidu_configs:
            tasks.append(asyncio.create_task(run_baidu_page(config, browser)))
        
        # 3. 添加 Sina 写入器任务 (虽然它已经在运行，但放入 gather 可以一起管理异常)
        tasks.append(sina_writer_task)

        try:
            # 并发运行所有任务
            await asyncio.gather(*tasks)
        except Exception as e:
            print(f"主程序异常: {e}")
        finally:
            print("正在关闭资源...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 程序已手动停止")