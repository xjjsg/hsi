# -*- coding: utf-8 -*-
"""
Alpha System Live - DEBUG MODE (话痨版)
--------------------------------------
功能：
1. [监控] 每一次 HTTP 请求都打印状态。
2. [监控] 每一笔 Tick 到达都打印计数 (不跳过)。
3. [诊断] 如果卡住，你能立刻看到是卡在网络还是卡在解析。
"""

import asyncio
import aiohttp
import torch
import numpy as np
import pandas as pd
import os
import sys
import json
import time
import warnings
from datetime import datetime, time as dt_time
from playwright.async_api import async_playwright

warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置
# ==========================================
CONFIG = {
    'MODEL_PATH': 'alpha_model_v8_stable.pth', 
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'RESAMPLE_FREQ': '3S',
    'LOOKBACK': 60,
    'CONF_THRESHOLD': 0.70,    
    'TRADE_COST': 0.0001,      
    'INITIAL_CAPITAL': 20000,  
    'MAX_POSITION': 0.8,
    'CLOSE_TIME': dt_time(14, 57, 0),
    
    # URL
    'SINA_URL': 'http://hq.sinajs.cn/list=hf_HSI,HKDCNY',
    'BAIDU_URL': 'https://gushitong.baidu.com/index/hk-HSI',
    'TENCENT_URL': 'http://qt.gtimg.cn/q=sz159920,sh513130',
    'MAIN_SYMBOL': 'sz159920',
    'AUX_SYMBOL': 'sh513130',
}

LIVE_CACHE = {
    "HSI": {"price": 0.0}, 
    "FUT": {"price": 0.0, "imb": 0.0},
    "FX": {"price": 0.92}
}

# ==========================================
# 2. 模型定义 (V8 保持不变)
# ==========================================
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class InceptionBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_chan, out_chan, 1), nn.LeakyReLU(0.01), nn.BatchNorm2d(out_chan))
        self.b2 = nn.Sequential(nn.Conv2d(in_chan, out_chan, 1), nn.LeakyReLU(0.01), nn.Conv2d(out_chan, out_chan, (3,1), padding=(1,0)), nn.LeakyReLU(0.01), nn.BatchNorm2d(out_chan))
        self.b3 = nn.Sequential(nn.Conv2d(in_chan, out_chan, 1), nn.LeakyReLU(0.01), nn.Conv2d(out_chan, out_chan, (5,1), padding=(2,0)), nn.LeakyReLU(0.01), nn.BatchNorm2d(out_chan))
        self.b4 = nn.Sequential(nn.MaxPool2d((3,1), stride=1, padding=(1,0)), nn.Conv2d(in_chan, out_chan, 1), nn.LeakyReLU(0.01), nn.BatchNorm2d(out_chan))
        self.se = SEBlock(out_chan * 4)
    def forward(self, x):
        return self.se(torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1))

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = float(hidden_size) ** 0.5
    def forward(self, x):
        last = x[:, -1, :].unsqueeze(1)
        scores = torch.bmm(self.query(last), self.key(x).transpose(1, 2)) / self.scale
        return torch.bmm(F.softmax(scores, dim=-1), self.value(x)).squeeze(1)

class HybridDeepLOB(nn.Module):
    def __init__(self, num_expert):
        super().__init__()
        c, m, l = 32, 64, 128
        self.compress = nn.Sequential(
            nn.Conv2d(1, c, (1, 2), stride=(1, 2)), nn.LeakyReLU(), nn.BatchNorm2d(c),
            nn.Conv2d(c, c, (4, 1), padding='same'), nn.LeakyReLU(), nn.BatchNorm2d(c),
            nn.Conv2d(c, c, (1, 2), stride=(1, 2)), nn.LeakyReLU(), nn.BatchNorm2d(c),
            nn.Conv2d(c, c, (1, 5), stride=(1, 5)), nn.LeakyReLU(), nn.BatchNorm2d(c)
        )
        self.inception1 = InceptionBlock(c, c)
        self.inception2 = InceptionBlock(128, 64)
        self.expert = nn.Sequential(nn.Linear(num_expert, m), nn.LeakyReLU(), nn.BatchNorm1d(m), nn.Dropout(0.2))
        self.lstm = nn.LSTM(256 + m, l, 2, batch_first=True, dropout=0.5)
        self.attention = TemporalAttention(l)
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Sequential(nn.Linear(l, 64), nn.LeakyReLU(), nn.Linear(64, 3))

    def forward(self, x_lob, x_exp):
        x = x_lob.unsqueeze(1)
        feat = self.inception2(self.inception1(self.compress(x))).squeeze(-1).permute(0, 2, 1)
        if feat.shape[1] != x_exp.shape[1]:
            feat = nn.functional.adaptive_avg_pool1d(feat.permute(0,2,1), x_exp.shape[1]).permute(0,2,1)
        B, T, F = x_exp.shape
        exp = self.expert(x_exp.reshape(-1, F)).reshape(B, T, -1)
        out, _ = self.lstm(torch.cat([feat, exp], dim=2))
        return self.head(self.dropout(self.attention(out)))

# ==========================================
# 3. 策略引擎 (Strategy) - 话痨版
# ==========================================
class StrategyEngine:
    def __init__(self):
        self.device = CONFIG['DEVICE']
        self.model = None
        self.features = ['meta_time', 'feat_micro_pressure', 'feat_oracle_basis', 
                         'feat_oracle_idx_mom', 'feat_oracle_fut_lead', 'feat_peer_diff']
        self.initial_capital = CONFIG['INITIAL_CAPITAL']
        self.cash = self.initial_capital
        self.holdings = 0
        self.trade_records = []
        self.done_for_day = False
        
        self.raw_buffer = pd.DataFrame()
        self.weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        self.warmup_cnt = 0
        
        self._load_model()

    def _load_model(self):
        print(f"🧠 加载模型: {CONFIG['MODEL_PATH']}")
        try:
            self.model = HybridDeepLOB(len(self.features)).to(self.device)
            state = torch.load(CONFIG['MODEL_PATH'], map_location=self.device)
            new_state = {}
            for k, v in state.items():
                k = k.replace('inc1.', 'inception1.').replace('inc2.', 'inception2.').replace('attn.', 'attention.')
                new_state[k] = v
            self.model.load_state_dict(new_state)
            self.model.eval()
            print("✅ 模型就绪")
        except Exception as e:
            print(f"❌ 模型错误: {e}")
            sys.exit(1)

    async def on_data(self, packet):
        if self.done_for_day: return

        # 1. 组装数据
        row = packet['main']
        row['peer_price'] = packet['aux']['price']
        row['peer_vol'] = packet['aux']['tick_vol']
        row['index_price'] = LIVE_CACHE['HSI']['price']
        row['fut_price'] = LIVE_CACHE['FUT']['price']
        
        df_row = pd.DataFrame([row])
        df_row['datetime'] = pd.to_datetime(datetime.now())
        df_row = df_row.set_index('datetime')
        
        # 2. 收盘检查
        current_time = datetime.now()
        if current_time.time() >= CONFIG['CLOSE_TIME']:
            print(f"\n⏰ 收盘清仓...")
            self.force_close(df_row)
            self.print_daily_report()
            self.done_for_day = True
            return

        # 3. 处理 Tick
        await self.process_tick(df_row)

    async def process_tick(self, df_snapshot):
        self.raw_buffer = pd.concat([self.raw_buffer, df_snapshot])
        if len(self.raw_buffer) > 300: self.raw_buffer = self.raw_buffer.iloc[-300:]
        
        # [DEBUG] 实时打印计数，不隐藏
        curr_len = len(self.raw_buffer)
        target_len = CONFIG['LOOKBACK'] + 20
        
        if curr_len < target_len:
            # 这里的 \r 可能会被 IDE 的输出覆盖，改为 print 看看是不是真的在动
            sys.stdout.write(f"\r⏳ [DEBUG] 预热进度: {curr_len}/{target_len} | 159920价格: {df_snapshot['price'].iloc[-1]} | 时间: {datetime.now().strftime('%H:%M:%S')}")
            sys.stdout.flush()
            self.warmup_cnt += 1
            return
        
        # 预热完成，开始计算
        if self.warmup_cnt > 0:
            print("\n✅ 预热完成，开始推理...")
            self.warmup_cnt = 0 # 标记为已完成

        # 计算因子
        try:
            df_factors = self._calc_realtime_factors(self.raw_buffer)
            for f in self.features:
                if f not in df_factors.columns: df_factors[f] = 0.0
                
            current = df_factors.iloc[-CONFIG['LOOKBACK']:]
            
            # 归一化
            exp_vals = current[self.features].values
            X_exp = np.nan_to_num((exp_vals - exp_vals.mean(0)) / (exp_vals.std(0)+1e-6)).astype(np.float32)
            
            lob_cols = [f'{s}{i}' for i in range(1,6) for s in ['bp','sp']] + \
                       [f'{s}{i}' for i in range(1,6) for s in ['bv','sv']]
            mid = current['mid'].values.reshape(-1, 1)
            lob_vals = current[lob_cols].values
            lob_vals[:, :10] = (lob_vals[:, :10] - mid) / (mid+1e-8) * 10000
            lob_vals[:, 10:] = np.log1p(lob_vals[:, 10:])
            X_lob = np.nan_to_num(lob_vals).astype(np.float32)
            
            # 推理
            t_lob = torch.tensor(X_lob).unsqueeze(0).to(self.device)
            t_exp = torch.tensor(X_exp).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                probs = torch.softmax(self.model(t_lob, t_exp), dim=1).cpu().numpy()[0]
                
            self._execute(probs, df_snapshot['price'].iloc[-1])
            
        except Exception as e:
            print(f"\n❌ 推理计算出错: {e}")

    def _execute(self, probs, price):
        p_hold, p_buy, p_sell = probs
        signal = 0; conf = 0.0
        if p_buy > CONFIG['CONF_THRESHOLD'] and p_buy > p_sell: signal=1; conf=p_buy
        elif p_sell > CONFIG['CONF_THRESHOLD'] and p_sell > p_buy: signal=2; conf=p_sell
        
        ts = datetime.now().strftime("%H:%M:%S")
        
        # [DEBUG] 无论是否有信号，都打印心跳，证明没卡死
        idx_p = LIVE_CACHE['HSI']['price']
        fut_p = LIVE_CACHE['FUT']['price']
        
        if signal != 0:
            action = "BUY" if signal == 1 else "SELL"
            scale = min((conf - CONFIG['CONF_THRESHOLD'])/(1 - CONFIG['CONF_THRESHOLD']), 1.0)
            invest = self.total_value(price) * CONFIG['MAX_POSITION'] * scale
            
            # 执行
            self._do_trade(signal, invest, price)
            print(f"\n⚡ [SIGNAL] {ts} | {action} | P:{price:.3f} | Conf:{conf:.2f} | Amt:{invest:.0f}")
        else:
            # 动态刷新状态栏
            sys.stdout.write(f"\r[{ts}] 👁️ 监控中... P:{price:.3f} Idx:{idx_p:.0f} Fut:{fut_p:.0f} | Buy:{p_buy:.2f} Sell:{p_sell:.2f}")
            sys.stdout.flush()

    def _do_trade(self, signal, amount, price):
        # 简化版交易逻辑
        diff = amount / price - self.holdings if signal==1 else -self.holdings # 简单的多/空切换
        if abs(diff) < 100: return
        
        cost = abs(diff) * price * CONFIG['TRADE_COST']
        if diff > 0:
            self.cash -= (diff * price + cost)
        else:
            self.cash += (abs(diff) * price - cost)
        self.holdings += diff
        self.trade_records.append({'time': datetime.now(), 'action': 'TRADE', 'price': price, 'qty': diff})

    def force_close(self, snapshot):
        if abs(self.holdings) < 1: return
        price = snapshot['price'].iloc[-1] # 简化用最新价
        cost = abs(self.holdings) * price * CONFIG['TRADE_COST']
        self.cash += (self.holdings * price - cost)
        self.holdings = 0
        print(f"✅ 清仓完成。")

    def print_daily_report(self):
        pnl = self.cash - self.initial_capital
        print(f"\n最终权益: {self.cash:.2f} | 盈亏: {pnl:.2f} | 交易: {len(self.trade_records)}")

    def total_value(self, price):
        return self.cash + self.holdings * price

    def _calc_realtime_factors(self, df):
        agg = {'price': 'last', 'tick_vol': 'sum', 'bp1':'last', 'sp1':'last'}
        for i in range(1,6):
            for s in ['bp','sp','bv','sv']: agg[f'{s}{i}'] = 'last'
        for c in ['index_price', 'fut_price', 'peer_price']:
            if c in df.columns: agg[c] = 'last'
            
        df_res = df.resample(CONFIG['RESAMPLE_FREQ']).agg(agg).dropna()
        mid = (df_res['bp1'] + df_res['sp1']) / 2
        df_res['mid'] = mid
        
        sec = df_res.index.hour * 3600 + df_res.index.minute * 60 + df_res.index.second
        df_res['meta_time'] = np.clip(np.where(sec<=41400, (sec-34200)/14400, 0.5+(sec-46800)/14400), 0, 1)
        
        wb = sum(df_res[f'bv{i}']*self.weights[i-1] for i in range(1,6))
        wa = sum(df_res[f'sv{i}']*self.weights[i-1] for i in range(1,6))
        df_res['feat_micro_pressure'] = (wb - wa) / (wb + wa + 1e-8)
        
        if 'index_price' in df_res.columns:
            df_res['feat_oracle_basis'] = (df_res['index_price'] - mid) / mid
            df_res['feat_oracle_idx_mom'] = df_res['index_price'].pct_change(2)
        if 'fut_price' in df_res.columns:
            df_res['feat_oracle_fut_lead'] = df_res['fut_price'].pct_change()
        if 'peer_price' in df_res.columns:
            df_res['feat_peer_diff'] = df_res['price'].pct_change() - df_res['peer_price'].pct_change()
            
        return df_res.fillna(0)

# ==========================================
# 4. 爬虫 (Verbose Mode)
# ==========================================
async def run_sina_worker():
    print("🔧 [Sina] 启动...")
    url = CONFIG['SINA_URL']
    headers = {"Referer": "https://finance.sina.com.cn/"}
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                # print("[DEBUG] Requesting Sina...") 
                async with session.get(url, headers=headers, timeout=5) as resp:
                    text = await resp.text(encoding='gbk')
                    parts = text.split('\n')
                    for p in parts:
                        if "hf_HSI" in p:
                            f = p.split(',')
                            if len(f)>6: LIVE_CACHE['FUT']['price'] = float(f[0].split('"')[-1])
            except Exception as e: 
                print(f"[Sina Error] {e}")
            await asyncio.sleep(0.5)

async def run_baidu_worker():
    print("🌍 [Baidu] 启动...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        def on_ws(ws):
            async def handle(payload):
                try:
                    data = json.loads(payload if isinstance(payload, str) else payload.decode('utf-8'))
                    price = float(data['data']['cur']['price'])
                    if price > 0: 
                        LIVE_CACHE['HSI']['price'] = price
                        # print(f"[DEBUG] Baidu Index: {price}")
                except: pass
            ws.on("framereceived", lambda x: asyncio.create_task(handle(x)))
        page.on("websocket", on_ws)
        try:
            await page.goto(CONFIG['BAIDU_URL'], timeout=60000)
            await asyncio.Future()
        except: pass
        finally: await browser.close()

class TencentWorker:
    def __init__(self, strategy):
        self.strategy = strategy
        self.last_snap = {}

    async def run(self):
        print("🐧 [Tencent] 启动...")
        url = CONFIG['TENCENT_URL']
        async with aiohttp.ClientSession() as session:
            while True:
                start = time.time()
                try:
                    # print("[DEBUG] Requesting Tencent...")
                    async with session.get(url, timeout=5) as resp:
                        text = await resp.text(encoding='gbk')
                        
                    data = self._parse_all(text)
                    if data:
                        await self.strategy.on_data(data)
                    else:
                        print("⚠️ [Tencent] 数据解析为空")
                        
                except Exception as e:
                    print(f"[Tencent Error] {e}")
                
                await asyncio.sleep(max(0, 1.0 - (time.time() - start)))

    def _parse_all(self, text):
        lines = text.strip().split(';')
        data = {}
        for line in lines:
            if CONFIG['MAIN_SYMBOL'] in line: data['main'] = self._parse_line(line, CONFIG['MAIN_SYMBOL'])
            elif CONFIG['AUX_SYMBOL'] in line: data['aux'] = self._parse_line(line, CONFIG['AUX_SYMBOL'])
        return data if 'main' in data and 'aux' in data else None

    def _parse_line(self, line, code):
        try:
            f = line.split('~')
            total_vol = float(f[6]) * 100
            tick_vol = max(0, total_vol - self.last_snap.get(code, total_vol))
            self.last_snap[code] = total_vol
            
            return {
                'price': float(f[3]), 'tick_vol': tick_vol,
                'bp1': float(f[9]), 'bv1': float(f[10])*100,
                'sp1': float(f[19]), 'sv1': float(f[20])*100,
                'bp2': float(f[11]), 'bv2': float(f[12])*100, 'bp3': float(f[13]), 'bv3': float(f[14])*100,
                'bp4': float(f[15]), 'bv4': float(f[16])*100, 'bp5': float(f[17]), 'bv5': float(f[18])*100,
                'sp2': float(f[21]), 'sv2': float(f[22])*100, 'sp3': float(f[23]), 'sv3': float(f[24])*100,
                'sp4': float(f[25]), 'sv4': float(f[26])*100, 'sp5': float(f[27]), 'sv5': float(f[28])*100,
            }
        except: return {}

# ==========================================
# Main
# ==========================================
async def main():
    print("=== Alpha System 实盘调试版 ===")
    strategy = StrategyEngine()
    tencent = TencentWorker(strategy)
    
    await asyncio.gather(
        run_sina_worker(),
        run_baidu_worker(),
        tencent.run()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已退出")