import asyncio
import aiohttp
import aiofiles
import csv
import os
import time
import sys
import json
import numpy as np
from datetime import datetime, time as dt_time  # 引入 time 类用于比较
from playwright.async_api import async_playwright, Browser

# ================= 配置区域 =================

CSV_HEADER_BASE = [
    "symbol", 
    "tx_server_time", "tx_local_time", "bd_server_time", "bd_local_time",
    "price", "iopv", "premium_rate", "index_price", "fx_rate", "sentiment",
    "tick_vol", "tick_amt", "tick_vwap", "interval_s",
    "bp1", "bv1", "bp2", "bv2", "bp3", "bv3", "bp4", "bv4", "bp5", "bv5",
    "sp1", "sv1", "sp2", "sv2", "sp3", "sv3", "sp4", "sv4", "sp5", "sv5"
]

CSV_HEADER_FUTURES = [
    "fut_local_time", "fut_tick_time", "fut_price", "fut_mid", 
    "fut_imb", "fut_delta_vol", "fut_pct"
]

ETF_CONFIG = {
    "sz159920": {
        "index_key": "HSI",    
        "future_key": "HSI",   
        "sentiment_key": "HSI",
        "has_futures": True,    
        "header": CSV_HEADER_BASE + CSV_HEADER_FUTURES
    },      
    "sh513130": {
        "index_key": "HZ2083", 
        "future_key": None,     
        "sentiment_key": "HSTECH",
        "has_futures": False,   
        "header": CSV_HEADER_BASE
    }    
}

BAIDU_CONFIGS = [
    {"url": "https://gushitong.baidu.com/index/hk-HSI",    "key": "HSI"},
    {"url": "https://gushitong.baidu.com/index/hk-HZ2083", "key": "HZ2083"}
]

SINA_FUTURES_CONFIG = ["hf_HSI"] 
SINA_FX_CONFIG = "HKDCNY"

HSI_WEIGHTS = {
    '00005': 8.0, '00700': 8.0, '09988': 8.0, '01299': 7.0, '00939': 5.5,
    '03690': 5.5, '00941': 4.0, '01398': 3.5, '00388': 3.0, '01211': 2.5
}

HSTECH_WEIGHTS = {
    '01810': 15.0, '03690': 13.0, '00700': 10.0, '09988': 8.0, '09618': 6.0,
    '01024': 6.0,  '02015': 5.0,  '09961': 4.0,  '00999': 3.5, '00981': 3.0
}

# ================= 全局缓存 =================

GLOBAL_INDEX_CACHE = {
    "HSI":    {"price": 0.0, "server_time": "N/A", "local_time": 0},
    "HZ2083": {"price": 0.0, "server_time": "N/A", "local_time": 0}
}

GLOBAL_FUTURES_CACHE = {
    "HSI": {
        "local_time": "N/A", "tick_time": "N/A", "price": 0.0, 
        "mid": 0.0, "imb": 0.0, "delta_vol": 0, "pct": 0.0
    }
}

GLOBAL_FX_CACHE = {"price": 0.92, "time": "N/A"}
GLOBAL_SENTIMENT_CACHE = {"HSI": 0.0, "HSTECH": 0.0}

# ==============================================================================
#                           MODULE: SENTIMENT ALGORITHM
# ==============================================================================

class SingleStockCalculator:
    def __init__(self, code):
        self.code = code
        self.last_vol = 0
        self.last_amt = 0
        self.vol_history = []
        self.score_ema = 0
        self.initialized = False

    def calculate(self, data_dict):
        curr_vol = data_dict['vol']
        curr_amt = data_dict['amount']
        curr_price = data_dict['price']
        
        if not self.initialized:
            self.last_vol = curr_vol
            self.last_amt = curr_amt
            self.initialized = True
            return 0
            
        delta_vol = curr_vol - self.last_vol
        delta_amt = curr_amt - self.last_amt
        self.last_vol = curr_vol
        self.last_amt = curr_amt
        
        if delta_vol <= 0:
            self.score_ema *= 0.95
            return self.score_ema
            
        vwap_3s = delta_amt / delta_vol
        diff_bp = (curr_price - vwap_3s) / vwap_3s * 10000
        score_momentum = np.clip(diff_bp * 1.5, -4, 4)
        
        score_structure = 0
        bid1 = data_dict['bid1']
        ask1 = data_dict['ask1']
        
        if bid1 > 0 and ask1 > 0:
            if curr_price >= ask1: score_structure = 2
            elif curr_price <= bid1: score_structure = -2
        else:
            score_momentum *= 1.2

        context_factor = 1.0
        day_low = data_dict['low']
        day_high = data_dict['high']
        
        if abs(curr_price - day_low) / day_low < 0.005 and score_momentum < 0:
            context_factor = 1.5
        elif abs(curr_price - day_high) / day_high < 0.005 and score_momentum > 0:
            context_factor = 1.2

        self.vol_history.append(delta_vol)
        if len(self.vol_history) > 20: self.vol_history.pop(0)
        avg_vol = np.mean(self.vol_history) + 100
        vol_ratio = min(delta_vol / avg_vol, 4.0)
        
        raw_score = (score_momentum + score_structure) * vol_ratio * context_factor
        self.score_ema = self.score_ema * 0.7 + raw_score * 0.3
        
        return self.score_ema

class SentimentWorker:
    def __init__(self):
        all_codes = set(list(HSI_WEIGHTS.keys()) + list(HSTECH_WEIGHTS.keys()))
        self.calculators = {code: SingleStockCalculator(code) for code in all_codes}
        self.hsi_norm_weights = self._normalize_weights(HSI_WEIGHTS)
        self.hstech_norm_weights = self._normalize_weights(HSTECH_WEIGHTS)

    def _normalize_weights(self, w_dict):
        total = sum(w_dict.values())
        return {k: v/total for k, v in w_dict.items()}

    def parse_stock_line(self, line):
        try:
            if '="' not in line: return None
            _, content = line.split('="')
            parts = content.strip('";\n').split('~')
            if len(parts) < 38: return None
            return {
                'code': parts[2],
                'price': float(parts[3]),
                'vol': float(parts[6]),
                'bid1': float(parts[9]),
                'ask1': float(parts[19]),
                'time': parts[31],
                'high': float(parts[34]),
                'low': float(parts[35]),
                'amount': float(parts[37]) if len(parts)>37 and float(parts[37]) > 10000 else float(parts[38])
            }
        except Exception:
            return None

    async def run(self):
        async with aiohttp.ClientSession() as session:
            codes_str = ",".join([f"r_hk{c}" for c in self.calculators.keys()])
            url = f"http://qt.gtimg.cn/q={codes_str}"

            while True:
                start_ts = time.time()
                try:
                    async with session.get(url) as resp:
                        text = await resp.text()
                    lines = text.strip().split(';')
                    stock_scores = {}
                    for line in lines:
                        if len(line) < 10: continue
                        data = self.parse_stock_line(line)
                        if data and data['code'] in self.calculators:
                            score = self.calculators[data['code']].calculate(data)
                            stock_scores[data['code']] = score

                    hsi_final = 0
                    for code, weight in self.hsi_norm_weights.items():
                        if code in stock_scores: hsi_final += stock_scores[code] * weight
                    
                    hstech_final = 0
                    for code, weight in self.hstech_norm_weights.items():
                        if code in stock_scores: hstech_final += stock_scores[code] * weight

                    GLOBAL_SENTIMENT_CACHE["HSI"] = round(np.tanh(hsi_final * 0.25) * 10, 4)
                    GLOBAL_SENTIMENT_CACHE["HSTECH"] = round(np.tanh(hstech_final * 0.25) * 10, 4)
                except Exception:
                    pass
                elapsed = time.time() - start_ts
                await asyncio.sleep(max(0, 1.0 - elapsed))

# ==============================================================================
#                           MODULE A: TENCENT ETF WORKER (时间限制版)
# ==============================================================================

class TencentETFWorker:
    def __init__(self):
        self.last_snapshot = {}
        # 定义时间段常量
        self.T1_START = dt_time(9, 29, 50)
        self.T1_END = dt_time(11, 30, 10)
        
        self.T2_INSTANT_H = 12
        self.T2_INSTANT_M = 0
        self.T2_INSTANT_S = 10
        
        self.T3_START = dt_time(12, 59, 50)
        self.T3_END = dt_time(15, 0, 10)

    def is_recording_time(self):
        """检查当前时间是否在允许记录的范围内"""
        now = datetime.now().time()
        
        # 1. 早盘
        if self.T1_START <= now <= self.T1_END:
            return True
            
        # 2. 午盘瞬时 (12:00:10)
        # 由于循环约1秒一次，只要秒数匹配即可
        if now.hour == self.T2_INSTANT_H and now.minute == self.T2_INSTANT_M and now.second == self.T2_INSTANT_S:
            return True
            
        # 3. 尾盘
        if self.T3_START <= now <= self.T3_END:
            return True
            
        return False

    def get_paths(self, symbol):
        """生成数据文件和错误日志的路径"""
        today_str = datetime.now().strftime("%Y-%m-%d")
        dir_path = os.path.join("data", symbol)
        os.makedirs(dir_path, exist_ok=True)
        
        data_file = os.path.join(dir_path, f"{symbol}-{today_str}.csv")
        error_file = os.path.join(dir_path, "error.txt")
        return data_file, error_file

    def parse_line(self, line: str):
        error_log = None 
        try:
            tx_local_ts_float = time.time()
            tx_local_time = int(tx_local_ts_float * 1000) 

            if '="' not in line: return None, None, None
            var_part, content = line.split('="')
            raw_symbol = var_part.split('_')[-1]
            f = content.strip('";\n').split('~')
            
            if len(f) < 80: return None, None, None

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
            
            # 【重要】无论是否记录，必须更新 last_snapshot，保证 vol 差值计算正确
            self.last_snapshot[raw_symbol] = {
                'vol': total_vol, 'amt': total_amt, 'local_ts_float': tx_local_ts_float
            }
            
            tick_vwap = (tick_amt / tick_vol) if tick_vol > 0 else price

            # --- 关联数据获取 ---
            config = ETF_CONFIG.get(raw_symbol, {})
            index_key = config.get("index_key", "")
            sentiment_key = config.get("sentiment_key", "")
            
            cached_idx = GLOBAL_INDEX_CACHE.get(index_key, {})
            index_price = cached_idx.get("price", 0.0)
            bd_server_time = cached_idx.get("server_time", "N/A")
            bd_local_time = cached_idx.get("local_time", 0)

            # 异常检查
            if index_price == 0 or index_price is None:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                error_log = f"[{timestamp}] Index=0 for {raw_symbol}. Raw Line: {line.strip()}\n"

            iopv = float(f[78]) if f[78] else 0.0
            premium_rate = (price - iopv) / iopv * 100 if iopv > 0 else 0.0
            
            fx_rate = GLOBAL_FX_CACHE.get("price", 0.0)
            sentiment_score = GLOBAL_SENTIMENT_CACHE.get(sentiment_key, 0.0)

            def safe_int_vol(idx): return int(f[idx]) * 100 

            row = [
                raw_symbol,
                tx_server_time, tx_local_time, bd_server_time, bd_local_time,
                price, iopv, round(premium_rate, 4), index_price, fx_rate, sentiment_score,
                int(tick_vol), int(tick_amt), round(tick_vwap, 4), round(interval_s, 3),
                f[9], safe_int_vol(10), f[11], safe_int_vol(12), 
                f[13], safe_int_vol(14), f[15], safe_int_vol(16), f[17], safe_int_vol(18),
                f[19], safe_int_vol(20), f[21], safe_int_vol(22), 
                f[23], safe_int_vol(24), f[25], safe_int_vol(26), f[27], safe_int_vol(28)
            ]

            fut_print_info = "" 
            if config.get("has_futures"):
                future_key = config.get("future_key", "")
                cached_fut = GLOBAL_FUTURES_CACHE.get(future_key, {})
                if not cached_fut:
                    cached_fut = {"local_time":"N/A", "tick_time":"N/A", "price":0, "mid":0, "imb":0, "delta_vol":0, "pct":0}
                row.extend([
                    cached_fut.get("local_time", "N/A"), cached_fut.get("tick_time", "N/A"),
                    cached_fut.get("price", 0), cached_fut.get("mid", 0), cached_fut.get("imb", 0),
                    cached_fut.get("delta_vol", 0), cached_fut.get("pct", 0)
                ])
                if cached_fut.get('price', 0) > 0:
                    fut_print_info = f" | Fut: {cached_fut.get('price')}"

            # 仅在需要记录的时间段才打印，避免刷屏（可选，这里保持打印以便监控）
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] {raw_symbol} | P:{price} | Idx:{index_price} | Prem:{premium_rate:.2f}%{fut_print_info}")

            return raw_symbol, row, error_log

        except Exception as e:
            # print(f"Parse Error: {e}")
            return None, None, None

    async def run(self):
        # 预先生成路径
        path_map = {} 
        for code, cfg in ETF_CONFIG.items():
            csv_path, err_path = self.get_paths(code)
            path_map[code] = {'csv': csv_path, 'err': err_path}
            
            if not os.path.exists(csv_path):
                async with aiofiles.open(csv_path, "w", newline="", encoding="utf-8") as f:
                    await f.write(",".join(cfg["header"]) + "\n")
        
        async with aiohttp.ClientSession() as session:
            codes_str = ",".join(ETF_CONFIG.keys())
            url = f"http://qt.gtimg.cn/q={codes_str}"
            
            while True:
                start_ts = time.time()
                try:
                    # 1. 获取并解析数据
                    async with session.get(url) as resp:
                        text = await resp.text(encoding='gbk')
                        
                    lines = text.strip().split(';')
                    
                    # 2. 检查是否在记录时间段
                    should_record = self.is_recording_time()

                    for line in lines:
                        if not line.strip(): continue
                        
                        # Parse总是执行，以保持 snapshot 状态更新 (Tick Volume计算正确)
                        symbol, row, error_msg = self.parse_line(line)
                        
                        if symbol and row and should_record:
                            paths = path_map.get(symbol)
                            if paths:
                                # 写入 CSV
                                async with aiofiles.open(paths['csv'], "a", newline="", encoding="utf-8") as f:
                                    await f.write(",".join(map(str, row)) + "\n")
                                
                                # 写入 Error Log (仅在记录时段内记录错误)
                                if error_msg:
                                    async with aiofiles.open(paths['err'], "a", encoding="utf-8") as f_err:
                                        await f_err.write(error_msg)

                except Exception:
                    pass
                
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
        
        if "cur" not in raw_data: return
        cur = raw_data["cur"]
        price = float(cur.get("price", 0))
        
        bd_server_time = "N/A"
        timestamp_sec = 0
        if "point" in raw_data and "realTimeStampMs" in raw_data["point"]:
            try:
                ms = int(raw_data["point"]["realTimeStampMs"])
                timestamp_sec = ms / 1000.0
            except: pass
        
        if timestamp_sec > 0:
            dt = datetime.fromtimestamp(timestamp_sec)
            bd_server_time = dt.strftime("%H:%M:%S")

        if price > 0:
            GLOBAL_INDEX_CACHE[key] = {
                "price": price,
                "server_time": bd_server_time,
                "local_time": bd_local_time
            }
    except Exception:
        pass

async def run_baidu_page(config: dict, browser: Browser):
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
    except Exception:
        pass
    finally:
        await context.close()

# ==============================================================================
#                           MODULE C: SINA FUTURES & FX WORKER
# ==============================================================================

class SinaDataWorker:
    def __init__(self):
        self.last_cum_vols = {} 
        self.headers = {
            "Referer": "https://finance.sina.com.cn/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"
        }

    def safe_float(self, val):
        try:
            if not val or not val.strip(): return 0.0
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    def parse_sina_data(self, text):
        lines = text.strip().split('\n')
        current_local_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        for line in lines:
            if '="' not in line: continue
            try:
                split_idx = line.find('="')
                if split_idx == -1: continue
                lhs = line[:split_idx]
                rhs = line[split_idx+2:] 
                data_str = rhs.strip('";')
                if not data_str: continue 
                f = data_str.split(',')

                if "hf_" in lhs:
                    code = lhs.split('_')[-1] 
                    if len(f) < 14: continue
                    price = self.safe_float(f[0])
                    bid1, ask1 = self.safe_float(f[2]), self.safe_float(f[3])
                    tick_time, prev = f[6], self.safe_float(f[7])
                    bid_vol, ask_vol = self.safe_float(f[10]), self.safe_float(f[11])
                    cum_vol = self.safe_float(f[14])

                    mid = (bid1 + ask1) / 2 if (bid1 > 0 and ask1 > 0) else price
                    total_depth = bid_vol + ask_vol
                    imb = (bid_vol - ask_vol) / total_depth if total_depth > 0 else 0.0
                    last_vol = self.last_cum_vols.get(code, cum_vol)
                    delta_vol = cum_vol - last_vol
                    self.last_cum_vols[code] = cum_vol 
                    pct = (price - prev) / prev * 100 if prev > 0 else 0.0

                    GLOBAL_FUTURES_CACHE[code] = {
                        "local_time": current_local_time, "tick_time": tick_time,
                        "price": price, "mid": round(mid, 2), "imb": round(imb, 2),
                        "delta_vol": int(delta_vol), "pct": round(pct, 2)
                    }
                elif "HKDCNY" in lhs:
                    if len(f) > 3:
                        last_price = self.safe_float(f[3])
                        if last_price > 0:
                            GLOBAL_FX_CACHE["price"] = last_price
                            GLOBAL_FX_CACHE["time"] = f[0]
            except Exception:
                pass

    async def run(self):
        codes = SINA_FUTURES_CONFIG + [SINA_FX_CONFIG]
        url = f"http://hq.sinajs.cn/list={','.join(codes)}"
        async with aiohttp.ClientSession() as session:
            while True:
                start_ts = time.time()
                try:
                    async with session.get(url, headers=self.headers) as resp:
                        text = await resp.text(encoding='gbk')
                        self.parse_sina_data(text)
                except Exception:
                    pass
                elapsed = time.time() - start_ts
                await asyncio.sleep(max(0, 0.5 - elapsed))

# ==============================================================================
#                                MAIN ENTRY POINT
# ==============================================================================

async def main():
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    print("启动各模块...")
    
    sentiment_worker = SentimentWorker()
    task_sentiment = asyncio.create_task(sentiment_worker.run())

    tencent_worker = TencentETFWorker()
    task_tencent = asyncio.create_task(tencent_worker.run())
    
    sina_worker = SinaDataWorker()
    task_sina = asyncio.create_task(sina_worker.run())

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        baidu_tasks = []
        for cfg in BAIDU_CONFIGS:
            baidu_tasks.append(asyncio.create_task(run_baidu_page(cfg, browser)))
        
        all_tasks = [task_tencent, task_sina, task_sentiment] + baidu_tasks
        try:
            print("系统运行中... (按 Ctrl+C 停止)")
            await asyncio.gather(*all_tasks)
        except Exception as e:
            print(f"Main Loop Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("程序已停止")