import asyncio
import aiohttp
import aiofiles
import csv
import os
import time
import sys
import json
import numpy as np  # 新增依赖
from datetime import datetime
from playwright.async_api import async_playwright, Browser

# ================= 配置区域 =================

# 基础表头 (新增 fx_rate 和 sentiment 字段)
CSV_HEADER_BASE = [
    "symbol", 
    "tx_server_time", "tx_local_time", "bd_server_time", "bd_local_time",
    "price", "iopv", "premium_rate", "index_price", "fx_rate", "sentiment",  # <--- 新增 sentiment
    "tick_vol", "tick_amt", "tick_vwap", "interval_s",
    "bp1", "bv1", "bp2", "bv2", "bp3", "bv3", "bp4", "bv4", "bp5", "bv5",
    "sp1", "sv1", "sp2", "sv2", "sp3", "sv3", "sp4", "sv4", "sp5", "sv5"
]

# 期货扩展表头
CSV_HEADER_FUTURES = [
    "fut_local_time", "fut_tick_time", "fut_price", "fut_mid", 
    "fut_imb", "fut_delta_vol", "fut_pct"
]

ETF_CONFIG = {
    "sz159920": {
        "file": "sz159920.csv", 
        "index_key": "HSI",    
        "future_key": "HSI",   
        "sentiment_key": "HSI", # <--- 绑定 HSI 情绪分
        "has_futures": True,    
        "header": CSV_HEADER_BASE + CSV_HEADER_FUTURES
    },      
    "sh513130": {
        "file": "sh513130.csv", 
        "index_key": "HZ2083", 
        "future_key": None,     
        "sentiment_key": "HSTECH", # <--- 绑定 HSTECH 情绪分
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

# --- 情绪算法相关配置 (来自 1.py) ---

# 1. 恒生指数 (159920) 前十大权重股
HSI_WEIGHTS = {
    '00005': 8.0, '00700': 8.0, '09988': 8.0, '01299': 7.0, '00939': 5.5,
    '03690': 5.5, '00941': 4.0, '01398': 3.5, '00388': 3.0, '01211': 2.5
}

# 2. 恒生科技指数 (513130) 前十大权重股
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

GLOBAL_FX_CACHE = {
    "price": 0.92,
    "time": "N/A"
}

# 新增：情绪分缓存
GLOBAL_SENTIMENT_CACHE = {
    "HSI": 0.0,
    "HSTECH": 0.0
}

# ==============================================================================
#                           MODULE: SENTIMENT ALGORITHM (from 1.py)
# ==============================================================================

class SingleStockCalculator:
    """ 单只股票 V3.0 纯数据算法 (不依赖挂单量) """
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
            
        # A. VWAP 博弈
        vwap_3s = delta_amt / delta_vol
        diff_bp = (curr_price - vwap_3s) / vwap_3s * 10000
        score_momentum = np.clip(diff_bp * 1.5, -4, 4)
        
        # B. 盘口侵略性
        score_structure = 0
        bid1 = data_dict['bid1']
        ask1 = data_dict['ask1']
        
        if bid1 > 0 and ask1 > 0:
            if curr_price >= ask1: score_structure = 2
            elif curr_price <= bid1: score_structure = -2
        else:
            score_momentum *= 1.2

        # C. 极值修正
        context_factor = 1.0
        day_low = data_dict['low']
        day_high = data_dict['high']
        
        if abs(curr_price - day_low) / day_low < 0.005 and score_momentum < 0:
            context_factor = 1.5
        elif abs(curr_price - day_high) / day_high < 0.005 and score_momentum > 0:
            context_factor = 1.2

        # D. 量能加权
        self.vol_history.append(delta_vol)
        if len(self.vol_history) > 20: self.vol_history.pop(0)
        avg_vol = np.mean(self.vol_history) + 100
        vol_ratio = min(delta_vol / avg_vol, 4.0)
        
        raw_score = (score_momentum + score_structure) * vol_ratio * context_factor
        self.score_ema = self.score_ema * 0.7 + raw_score * 0.3
        
        return self.score_ema

class SentimentWorker:
    def __init__(self):
        # 初始化计算器
        all_codes = set(list(HSI_WEIGHTS.keys()) + list(HSTECH_WEIGHTS.keys()))
        self.calculators = {code: SingleStockCalculator(code) for code in all_codes}
        
        # 归一化权重
        self.hsi_norm_weights = self._normalize_weights(HSI_WEIGHTS)
        self.hstech_norm_weights = self._normalize_weights(HSTECH_WEIGHTS)

    def _normalize_weights(self, w_dict):
        total = sum(w_dict.values())
        return {k: v/total for k, v in w_dict.items()}

    def parse_stock_line(self, line):
        try:
            # 兼容处理 v_r_hk00700="1~00700~..."
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
            # 构造请求 URL (港股前缀 r_hk)
            codes_str = ",".join([f"r_hk{c}" for c in self.calculators.keys()])
            url = f"http://qt.gtimg.cn/q={codes_str}"

            while True:
                start_ts = time.time()
                try:
                    async with session.get(url) as resp:
                        text = await resp.text()
                    
                    lines = text.strip().split(';')
                    stock_scores = {}

                    # 1. 计算单股得分
                    for line in lines:
                        if len(line) < 10: continue
                        data = self.parse_stock_line(line)
                        if data and data['code'] in self.calculators:
                            score = self.calculators[data['code']].calculate(data)
                            stock_scores[data['code']] = score

                    # 2. 合成指数情绪
                    hsi_final = 0
                    for code, weight in self.hsi_norm_weights.items():
                        if code in stock_scores:
                            hsi_final += stock_scores[code] * weight
                    
                    hstech_final = 0
                    for code, weight in self.hstech_norm_weights.items():
                        if code in stock_scores:
                            hstech_final += stock_scores[code] * weight

                    # 3. Tanh 映射并更新全局缓存
                    GLOBAL_SENTIMENT_CACHE["HSI"] = round(np.tanh(hsi_final * 0.25) * 10, 4)
                    GLOBAL_SENTIMENT_CACHE["HSTECH"] = round(np.tanh(hstech_final * 0.25) * 10, 4)

                except Exception as e:
                    # print(f"Sentiment Worker Error: {e}")
                    pass
                
                # 保持约 3秒一次的更新频率 (与 1.py 保持一致，或更快)
                elapsed = time.time() - start_ts
                await asyncio.sleep(max(0, 1.0 - elapsed))

# ==============================================================================
#                           MODULE A: TENCENT ETF WORKER
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
            
            # --- 关联数据获取 ---
            config = ETF_CONFIG.get(raw_symbol, {})
            index_key = config.get("index_key", "")
            sentiment_key = config.get("sentiment_key", "") # 获取绑定的情绪key
            
            # 1. 指数
            cached_idx = GLOBAL_INDEX_CACHE.get(index_key, {})
            index_price = cached_idx.get("price", 0.0)
            bd_server_time = cached_idx.get("server_time", "N/A")
            bd_local_time = cached_idx.get("local_time", 0)

            # 2. 汇率
            fx_rate = GLOBAL_FX_CACHE.get("price", 0.0)

            # 3. 情绪分 (新增)
            sentiment_score = GLOBAL_SENTIMENT_CACHE.get(sentiment_key, 0.0)

            def safe_int_vol(idx):
                return int(f[idx]) * 100 

            row = [
                raw_symbol,
                tx_server_time, tx_local_time, bd_server_time, bd_local_time,
                price, iopv, round(premium_rate, 4), index_price, fx_rate, sentiment_score, # <--- 填入 sentiment
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
                    cached_fut.get("local_time", "N/A"),
                    cached_fut.get("tick_time", "N/A"),
                    cached_fut.get("price", 0),
                    cached_fut.get("mid", 0),
                    cached_fut.get("imb", 0),
                    cached_fut.get("delta_vol", 0),
                    cached_fut.get("pct", 0)
                ])
                
                if cached_fut.get('price', 0) > 0:
                    fut_print_info = f" | Fut: {cached_fut.get('price')} (Imb: {cached_fut.get('imb')})"

            # 打印包含汇率和情绪的信息
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] {raw_symbol} | P: {price} | Sent: {sentiment_score:>5.2f} | FX: {fx_rate} | Prem: {premium_rate:.2f}%{fut_print_info}")

            return raw_symbol, row
        except Exception as e:
            # print(f"Parse Error: {e}")
            return None, None

    async def run(self):
        # 初始化CSV文件
        for code, cfg in ETF_CONFIG.items():
            if not os.path.exists(cfg["file"]):
                with open(cfg["file"], "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(cfg["header"])
        
        async with aiohttp.ClientSession() as session:
            codes_str = ",".join(ETF_CONFIG.keys())
            url = f"http://qt.gtimg.cn/q={codes_str}"
            
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

                # 区分是期货还是汇率
                # 1. 期货 (hf_...)
                if "hf_" in lhs:
                    code = lhs.split('_')[-1] # "HSI"
                    if len(f) < 14: continue

                    price = self.safe_float(f[0])
                    bid1  = self.safe_float(f[2])
                    ask1  = self.safe_float(f[3])
                    tick_time = f[6]
                    prev  = self.safe_float(f[7])
                    bid_vol = self.safe_float(f[10])
                    ask_vol = self.safe_float(f[11])
                    cum_vol = self.safe_float(f[14])

                    mid = (bid1 + ask1) / 2 if (bid1 > 0 and ask1 > 0) else price
                    total_depth = bid_vol + ask_vol
                    imb = (bid_vol - ask_vol) / total_depth if total_depth > 0 else 0.0

                    last_vol = self.last_cum_vols.get(code, cum_vol)
                    delta_vol = cum_vol - last_vol
                    self.last_cum_vols[code] = cum_vol 

                    pct = (price - prev) / prev * 100 if prev > 0 else 0.0

                    GLOBAL_FUTURES_CACHE[code] = {
                        "local_time": current_local_time,
                        "tick_time": tick_time,
                        "price": price,
                        "mid": round(mid, 2),
                        "imb": round(imb, 2),
                        "delta_vol": int(delta_vol),
                        "pct": round(pct, 2)
                    }

                # 2. 汇率 (HKDCNY)
                elif "HKDCNY" in lhs:
                    if len(f) > 3:
                        last_price = self.safe_float(f[3])
                        tick_time = f[0]
                        if last_price > 0:
                            GLOBAL_FX_CACHE["price"] = last_price
                            GLOBAL_FX_CACHE["time"] = tick_time

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
    
    # 1. 启动情绪计算 Worker
    sentiment_worker = SentimentWorker()
    task_sentiment = asyncio.create_task(sentiment_worker.run())

    # 2. 启动 ETF 记录 Worker
    tencent_worker = TencentETFWorker()
    task_tencent = asyncio.create_task(tencent_worker.run())
    
    # 3. 启动新浪数据 Worker (期货+汇率)
    sina_worker = SinaDataWorker()
    task_sina = asyncio.create_task(sina_worker.run())

    # 4. 启动 Playwright (百度指数)
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