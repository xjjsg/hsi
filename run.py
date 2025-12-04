import asyncio
import sys
import torch
import numpy as np
import pandas as pd
import time
import aiohttp
from collections import deque
from datetime import datetime, time as dt_time
from playwright.async_api import async_playwright
import warnings

# === 引入依赖 ===
try:
    from getdata import (
        SinaDataWorker, SentimentWorker, 
        BAIDU_CONFIGS, run_baidu_page, ETF_CONFIG,
        GLOBAL_INDEX_CACHE, GLOBAL_FUTURES_CACHE, GLOBAL_FX_CACHE, GLOBAL_SENTIMENT_CACHE
    )
    from getfactor import (
        CONFIG, Direction, ManualFactorGenerator
    )
except ImportError as e:
    print(f"❌ 缺少必要文件: {e}")
    sys.exit(1)

warnings.filterwarnings('ignore')

# === 全局内存缓存 ===
MEMORY_CACHE = {
    CONFIG['MAIN_SYMBOL']: deque(maxlen=300),
    CONFIG['AUX_SYMBOL']:  deque(maxlen=300)
}
HEADERS = {code: cfg['header'] for code, cfg in ETF_CONFIG.items()}

# === [核心修复] 纯净版 Worker (强制数值转换) ===
class SilentTencentWorker:
    def __init__(self):
        self.last_snapshot = {}

    def parse_line_silent(self, line: str):
        try:
            tx_local_ts_float = time.time()
            tx_local_time = int(tx_local_ts_float * 1000) 

            if '="' not in line: return None, None
            var_part, content = line.split('="')
            raw_symbol = var_part.split('_')[-1]
            f = content.strip('";\n').split('~')
            
            if len(f) < 80: return None, None

            # --- 基础字段 ---
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

            # --- 关联数据 ---
            config = ETF_CONFIG.get(raw_symbol, {})
            index_key = config.get("index_key", "")
            sentiment_key = config.get("sentiment_key", "")
            
            cached_idx = GLOBAL_INDEX_CACHE.get(index_key, {})
            index_price = cached_idx.get("price", 0.0)
            bd_server_time = cached_idx.get("server_time", "N/A")
            bd_local_time = cached_idx.get("local_time", 0)

            iopv = float(f[78]) if f[78] else 0.0
            premium_rate = (price - iopv) / iopv * 100 if iopv > 0 else 0.0
            
            fx_rate = GLOBAL_FX_CACHE.get("price", 0.0)
            sentiment_score = GLOBAL_SENTIMENT_CACHE.get(sentiment_key, 0.0)

            def safe_int_vol(idx): return int(f[idx]) * 100 

            # 【关键修复】这里全部强制转 float/int，不再保留字符串
            row = [
                raw_symbol,
                tx_server_time, tx_local_time, bd_server_time, bd_local_time,
                price, iopv, round(premium_rate, 4), index_price, fx_rate, sentiment_score,
                int(tick_vol), int(tick_amt), round(tick_vwap, 4), round(interval_s, 3),
                float(f[9]), safe_int_vol(10), float(f[11]), safe_int_vol(12),  # bp1
                float(f[13]), safe_int_vol(14), float(f[15]), safe_int_vol(16), float(f[17]), safe_int_vol(18),
                float(f[19]), safe_int_vol(20), float(f[21]), safe_int_vol(22), # sp1
                float(f[23]), safe_int_vol(24), float(f[25]), safe_int_vol(26), float(f[27]), safe_int_vol(28)
            ]

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

            return raw_symbol, row

        except Exception:
            return None, None

    async def run(self):
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
                        symbol, row = self.parse_line_silent(line)
                        if symbol and row and symbol in MEMORY_CACHE:
                            MEMORY_CACHE[symbol].append(row)

                except Exception:
                    pass
                elapsed = time.time() - start_ts
                await asyncio.sleep(max(0, 1.0 - elapsed))

# === 订单执行器 ===
class OrderExecutor:
    def __init__(self, initial_cash=100000.0):
        self.initial_capital = initial_cash
        self.cash = initial_cash
        self.position = 0
        self.entry_price = 0.0
        self.entry_cost_total = 0.0 
        
        self.fixed_cost = 10.0
        self.trade_val = 100000.0 
        
        self.trade_records = [] 
        self.last_buy_time = None

    def generate_order(self, score, current_price, ask1, bid1, is_noon_shock=False):
        direction = 1 if score > 0 else -1
        order_type = "LIMIT"
        order_price = 0.0
        urgency = "LOW"
        abs_score = abs(score)
        
        if is_noon_shock or abs_score > 0.7:
            urgency = "HIGH"
            order_type = "MARKET_TAKER"
            order_price = ask1 if direction == 1 else bid1
        elif abs_score > 0.4:
            urgency = "MID"
            order_type = "LIMIT_MAKER"
            order_price = bid1 if direction == 1 else ask1
        else:
            return None 

        vol = int(self.trade_val / order_price / 100) * 100
        if vol == 0: return None

        return {
            'action': 'OPEN' if self.position == 0 else 'CLOSE',
            'direction': direction,
            'price': order_price,
            'vol': vol,
            'type': order_type,
            'urgency': urgency
        }

    def match_order(self, order, market_snapshot):
        if not order: return False
        
        ask1 = market_snapshot['sp1']
        bid1 = market_snapshot['bp1']
        current_ts = market_snapshot.name 
        
        is_filled = False
        fill_price = order['price']
        
        if 'TAKER' in order['type'] or 'FORCE' in order['type']:
            is_filled = True
            fill_price = order['price'] 
        elif 'MAKER' in order['type']:
            if order['direction'] == 1: 
                if ask1 <= order['price']:
                    is_filled = True
                    fill_price = ask1 
            else: 
                if bid1 >= order['price']:
                    is_filled = True
                    fill_price = bid1

        if is_filled:
            self._execute_accounting(order, fill_price, current_ts)
            return True
        return False

    def _execute_accounting(self, order, price, ts):
        cost = self.fixed_cost
        val = order['vol'] * price
        
        if order['direction'] == 1: # BUY
            actual_cost = val + cost
            self.cash -= actual_cost
            self.position += order['vol']
            self.entry_price = price
            self.entry_cost_total = actual_cost
            self.last_buy_time = ts
            print(f"🔵 [成交] BUY  {order['vol']} @ {price:.3f} | 成本: {cost} | 类型: {order['type']}")
            
        else: # SELL
            actual_revenue = val - cost
            self.cash += actual_revenue
            net_profit = actual_revenue - self.entry_cost_total
            ret_pct = (net_profit / self.entry_cost_total) * 100
            
            self.trade_records.append({
                'buy_time': self.last_buy_time,
                'sell_time': ts,
                'net_pnl': net_profit,
                'ret_pct': ret_pct,
                'hold_time': (ts - self.last_buy_time).total_seconds() if self.last_buy_time else 0
            })
            
            color = "\033[91m" if net_profit > 0 else "\033[92m"
            print(f"🔴 [成交] SELL {order['vol']} @ {price:.3f} | 净利: {color}{net_profit:.2f}元 ({ret_pct:.2f}%)\033[0m")
            
            self.position = 0
            self.entry_price = 0
            self.entry_cost_total = 0

    def print_daily_report(self):
        print("\n" + "="*50)
        print(f"📊 每日交易报告 | {datetime.now().strftime('%Y-%m-%d')}")
        print("="*50)
        
        if not self.trade_records:
            print("⚠️ 今日无成交记录")
            return

        df_trades = pd.DataFrame(self.trade_records)
        total_trades = len(df_trades)
        wins = df_trades[df_trades['net_pnl'] > 0]
        losses = df_trades[df_trades['net_pnl'] <= 0]
        
        total_pnl = df_trades['net_pnl'].sum()
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        
        avg_win = wins['net_pnl'].mean() if not wins.empty else 0
        avg_loss = abs(losses['net_pnl'].mean()) if not losses.empty else 0
        pl_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        capital_ret = (total_pnl / self.initial_capital) * 100
        
        print(f"💰 最终权益:     {self.cash:.2f}")
        print(f"📈 累计盈亏:     {total_pnl:+.2f} 元")
        print(f"🚀 资金收益率:   {capital_ret:+.3f}%")
        print(f"-"*30)
        print(f"🎲 交易笔数:     {total_trades}")
        print(f"🏆 胜率:         {win_rate:.1f}%")
        print(f"⚖️ 盈亏比:       {pl_ratio:.2f}")
        print(f"⏱️ 平均持仓:     {df_trades['hold_time'].mean():.1f} 秒")
        print(f"📉 最大亏损:     {df_trades['net_pnl'].min():.2f} 元")
        print("="*50 + "\n")

# === 策略逻辑 ===
class SimulationStrategy:
    def __init__(self):
        print("🚀 [SimTrader] 模拟盘启动...")
        self.device = CONFIG['DEVICE']
        self.lookback = CONFIG['MAX_LOOKBACK']
        
        try:
            artifact = torch.load(CONFIG['ARTIFACT_NAME'], map_location=self.device, weights_only=False)
        except Exception:
            artifact = torch.load(CONFIG['ARTIFACT_NAME'], map_location=self.device)
            
        self.model = Direction(artifact['meta']['input_feature_count'], d_model=256, num_layers=6).to(self.device)
        self.model.load_state_dict(artifact['models']['direction_state_dict'])
        self.model.eval()
        self.scaler = artifact['models']['direction_scaler']
        self.manual_gen = ManualFactorGenerator()
        self.features = artifact['features']['input_names']
        
        self.executor = OrderExecutor(initial_cash=100000.0)
        self.last_ts = None
        self.report_printed = False

    async def run_loop(self):
        while True:
            now = datetime.now().time()
            if now > dt_time(15, 5) and not self.report_printed:
                self.executor.print_daily_report()
                self.report_printed = True
                print("🏁 市场已收盘。")
                sys.exit(0)

            df = self._get_dataframe()
            if df is not None and len(df) > self.lookback:
                curr_ts = df.index[-1]
                if curr_ts != self.last_ts:
                    self.last_ts = curr_ts
                    await self.on_tick(df)
            
            await asyncio.sleep(0.5)

    def _get_dataframe(self):
        main_code = CONFIG['MAIN_SYMBOL']
        aux_code = CONFIG['AUX_SYMBOL']
        
        if len(MEMORY_CACHE[main_code]) < 50 or len(MEMORY_CACHE[aux_code]) < 50:
            return None
            
        df_m = pd.DataFrame(list(MEMORY_CACHE[main_code]), columns=HEADERS[main_code])
        df_a = pd.DataFrame(list(MEMORY_CACHE[aux_code]), columns=HEADERS[aux_code])
        
        # 【关键修复】再次强制清洗，防止漏网之鱼
        df_m = df_m.apply(pd.to_numeric, errors='ignore')
        df_a = df_a.apply(pd.to_numeric, errors='ignore')

        for df in [df_m, df_a]:
            df['datetime'] = pd.to_datetime(df['tx_local_time'], unit='ms')
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            df = df[~df.index.duplicated(keep='last')]
        
        df_m = df_m.resample('3s').last().ffill()
        df_a = df_a.resample('3s').last().ffill()
        
        if 'bp1' not in df_m.columns: return None
        
        # 现在这里是安全的
        df_m['mid_price'] = (df_m['bp1'] + df_m['sp1']) / 2
        df_m['log_ret'] = np.log(df_m['mid_price'] / df_m['mid_price'].shift(1)).fillna(0)
        df_a = df_a.add_prefix('ctx_')
        
        return df_m.join(df_a, how='inner').dropna()

    async def on_tick(self, df):
        # 1. 推理
        df_feat = self.manual_gen.process(df)
        df_final = df.join(df_feat, how='inner')
        
        input_data = df_final.iloc[-self.lookback:]
        raw = np.nan_to_num(input_data[self.features].values, nan=0.0)
        norm = self.scaler.transform(raw)
        tensor = torch.FloatTensor(norm).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prob = torch.softmax(self.model(tensor), dim=1)
            score = (prob[0, 1] - prob[0, 2]).item()
            
        # 2. 状态与逻辑
        snapshot = df_final.iloc[-1]
        price = snapshot['price']
        ask1 = snapshot['sp1']
        bid1 = snapshot['bp1']
        ts_time = snapshot.name.time()
        
        is_noon = dt_time(13,0) <= ts_time <= dt_time(13,5)
        
        # 14:57 尾盘清仓
        if ts_time >= dt_time(14, 57):
            if self.executor.position > 0:
                print(f"⏰ [尾盘] 强制清仓")
                order = {
                    'action': 'CLOSE', 'direction': -1, 'vol': self.executor.position,
                    'price': bid1, 'type': 'MARKET_FORCE', 'urgency': 'HIGH'
                }
                self.executor.match_order(order, snapshot)
            return

        # 信号执行
        target_dir = 0
        if self.executor.position == 0 and score > 0: target_dir = 1
        elif self.executor.position > 0 and score < 0: target_dir = -1
        
        if target_dir != 0:
            order = self.executor.generate_order(score, price, ask1, bid1, is_noon)
            if order:
                if order['action'] == 'OPEN' and self.executor.position == 0:
                    self.executor.match_order(order, snapshot)
                elif order['action'] == 'CLOSE' and self.executor.position > 0:
                    order['vol'] = self.executor.position
                    self.executor.match_order(order, snapshot)

        # 状态栏打印
        color = "\033[91m" if score > 0 else "\033[92m"
        print(f"\r[{ts_time}] P:{price:.3f} | S:{color}{score:+.3f}\033[0m | Pos:{self.executor.position}   ", end="")

# === 启动入口 ===
async def main():
    tencent = SilentTencentWorker()
    sina = SinaDataWorker()
    sentiment = SentimentWorker()
    strategy = SimulationStrategy()
    
    # 使用标准 asyncio.run 兼容 Playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        baidu_tasks = [run_baidu_page(cfg, browser) for cfg in BAIDU_CONFIGS]
        
        try:
            await asyncio.gather(
                tencent.run(),
                sina.run(),
                sentiment.run(),
                strategy.run_loop(),
                *baidu_tasks
            )
        except asyncio.CancelledError:
            pass
        finally:
            if not strategy.report_printed:
                strategy.executor.print_daily_report()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass