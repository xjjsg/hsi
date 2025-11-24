import asyncio
import aiohttp
import aiofiles
import csv
import os
import time
import sys
import json
from datetime import datetime
from playwright.async_api import async_playwright, Browser

ETF_CONFIG = {
    "sz159920": {"file": "sz159920.csv", "index_key": "HSI"},      
    "sh513130": {"file": "sh513130.csv",     "index_key": "HZ2083"}    
}

BAIDU_CONFIGS = [
    {"url": "https://gushitong.baidu.com/index/hk-HSI",    "key": "HSI"},
    {"url": "https://gushitong.baidu.com/index/hk-HZ2083", "key": "HZ2083"}
]

CSV_HEADER = [
    "symbol", 
    "tx_server_time", "tx_local_time", "bd_server_time", "bd_local_time",
    "price", "iopv", "premium_rate", "index_price",
    "tick_vol", "tick_amt", "tick_vwap", "interval_s",
    "bp1", "bv1", "bp2", "bv2", "bp3", "bv3", "bp4", "bv4", "bp5", "bv5",
    "sp1", "sv1", "sp2", "sv2", "sp3", "sv3", "sp4", "sv4", "sp5", "sv5"
]

GLOBAL_INDEX_CACHE = {
    "HSI":    {"price": 0.0, "server_time": "N/A", "local_time": 0},
    "HZ2083": {"price": 0.0, "server_time": "N/A", "local_time": 0}
}

# ==============================================================================
#                           MODULE A: TENCENT ETF PARSER
# ==============================================================================

class TencentETFWorker:
    def __init__(self):
        self.last_snapshot = {}

    def parse_line(self, line: str):
        try:
            tx_local_ts_float = time.time()
            tx_local_time = int(tx_local_ts_float * 1000) 

            if '="' not in line: return None
            var_part, content = line.split('="')
            raw_symbol = var_part.split('_')[-1]
            f = content.strip('";\n').split('~')
            
            if len(f) < 80: return None 

            t_str = f[30]
            tx_server_time = f"{t_str[8:10]}:{t_str[10:12]}:{t_str[12:14]}"
            
            price = float(f[3])
            
            total_vol = float(f[6]) * 100 
            try:
                total_amt = float(f[35].split('/')[2])
            except:
                total_amt = float(f[37]) * 10000
            
            tick_vol = 0
            tick_amt = 0
            interval_s = 0.0
            
            if raw_symbol in self.last_snapshot:
                last = self.last_snapshot[raw_symbol]
                tick_vol = total_vol - last['vol']
                tick_amt = total_amt - last['amt']
                interval_s = tx_local_ts_float - last['local_ts_float']
            
            self.last_snapshot[raw_symbol] = {
                'vol': total_vol, 'amt': total_amt, 'local_ts_float': tx_local_ts_float
            }
            
            tick_vwap = (tick_amt / tick_vol) if tick_vol > 0 else price

            iopv = float(f[78]) if f[78] else 0.0
            premium_rate = (price - iopv) / iopv * 100 if iopv > 0 else 0.0
            
            index_key = ETF_CONFIG.get(raw_symbol, {}).get("index_key", "")
            cached_idx = GLOBAL_INDEX_CACHE.get(index_key, {})
            
            index_price = cached_idx.get("price", 0.0)
            bd_server_time = cached_idx.get("server_time", "N/A")
            bd_local_time = cached_idx.get("local_time", 0)

            def safe_int_vol(idx):
                return int(f[idx]) * 100 

            row = [
                raw_symbol,
                tx_server_time, tx_local_time, bd_server_time, bd_local_time,
                price, iopv, round(premium_rate, 4), index_price,
                int(tick_vol), int(tick_amt), round(tick_vwap, 4), round(interval_s, 3),
                f[9], safe_int_vol(10), f[11], safe_int_vol(12), 
                f[13], safe_int_vol(14), f[15], safe_int_vol(16), f[17], safe_int_vol(18),
                f[19], safe_int_vol(20), f[21], safe_int_vol(22), 
                f[23], safe_int_vol(24), f[25], safe_int_vol(26), f[27], safe_int_vol(28)
            ]
            return raw_symbol, row
        except Exception:
            return None, None

    async def run(self):
        for code, cfg in ETF_CONFIG.items():
            if not os.path.exists(cfg["file"]):
                with open(cfg["file"], "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(CSV_HEADER)
        
        async with aiohttp.ClientSession() as session:
            codes_str = ",".join(ETF_CONFIG.keys())
            url = f"http://qt.gtimg.cn/q={codes_str}"
            
            print(f"[Tencent] 启动监控 ETF: {codes_str}...")

            while True:
                start_ts = time.time()
                try:
                    async with session.get(url) as resp:
                        text = await resp.text(encoding='gbk')
                        
                    lines = text.strip().split(';')
                    for line in lines:
                        if not line.strip(): continue
                        
                        symbol, row = self.parse_line(line)
                        if symbol and row:
                            target_file = ETF_CONFIG[symbol]["file"]
                            async with aiofiles.open(target_file, "a", newline="", encoding="utf-8") as f:
                                line_str = ",".join(map(str, row)) + "\n"
                                await f.write(line_str)

                            print(f"\r[Data] {symbol} TxTime:{row[1]} BdTime:{row[3]} Prem:{row[7]}%   ", end="")

                except Exception as e:
                    print(f"\n[Tencent] Error: {e}")
                
                elapsed = time.time() - start_ts
                await asyncio.sleep(max(0, 1.0 - elapsed))

# ==============================================================================
#                           MODULE B: BAIDU INDEX WORKER
# ==============================================================================

async def baidu_update_cache(msg: str, key: str):
    try:
        bd_local_time = int(time.time() * 1000)
        
        data = json.loads(msg)
        if "data" not in data: return
        raw_data = data["data"]
        
        # 1. 获取价格 (仍在 cur 中)
        if "cur" not in raw_data: return
        cur = raw_data["cur"]
        price = float(cur.get("price", 0))
        
        bd_server_time = "N/A"
        timestamp_sec = 0

        #优先使用 point.realTimeStampMs
        if "point" in raw_data and "realTimeStampMs" in raw_data["point"]:
            try:
                ms = int(raw_data["point"]["realTimeStampMs"])
                timestamp_sec = ms / 1000.0
            except:
                pass
        
        # 将时间戳转换为 HH:MM:SS 格式
        if timestamp_sec > 0:
            dt = datetime.fromtimestamp(timestamp_sec)
            bd_server_time = dt.strftime("%H:%M:%S")

        # 3. 更新缓存
        if price > 0:
            GLOBAL_INDEX_CACHE[key] = {
                "price": price,
                "server_time": bd_server_time,
                "local_time": bd_local_time
            }

    except Exception as e:
        print(f"\n[Baidu Parse Error] {e}")

async def run_baidu_page(config: dict, browser: Browser):
    print(f"[Baidu] 启动监听: {config['key']}")
    context = await browser.new_context()
    page = await context.new_page()

    def on_web_socket(ws):
        async def handle_msg(payload):
            content = payload.decode('utf-8') if isinstance(payload, bytes) else payload
            await baidu_update_cache(content, config['key'])
        ws.on("framereceived", lambda payload: asyncio.create_task(handle_msg(payload)))

    page.on("websocket", on_web_socket)

    try:
        await page.goto(config['url'], wait_until="domcontentloaded", timeout=60000)
        await asyncio.Future()
    except Exception as e:
        print(f"[Baidu] Page Error ({config['key']}): {e}")
    finally:
        await context.close()

# ==============================================================================
#                                MAIN ENTRY POINT
# ==============================================================================

async def main():
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    tencent_worker = TencentETFWorker()
    task_tencent = asyncio.create_task(tencent_worker.run())
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        baidu_tasks = []
        for cfg in BAIDU_CONFIGS:
            baidu_tasks.append(asyncio.create_task(run_baidu_page(cfg, browser)))
        
        all_tasks = [task_tencent] + baidu_tasks
        try:
            await asyncio.gather(*all_tasks)
        except Exception as e:
            print(f"Main Loop Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("程序已停止")