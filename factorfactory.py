# -*- coding: utf-8 -*-
"""
文件名: factorfactory_pro.py
版本: Pro (Dual Asset Support)
功能: 
    1. 同时处理 159920 (主) 和 513130 (辅)
    2. 计算传统的单品种微观因子
    3. [新增] 计算跨品种博弈因子 (Cross-Asset Alphas)
"""

import pandas as pd
import numpy as np
from datetime import time

class FactorFactoryPro:
    def __init__(self, main_file, aux_file, main_code='159920', aux_code='513130', resample_rule='3s'):
        self.main_file = main_file
        self.aux_file = aux_file
        self.main_code = main_code
        self.aux_code = aux_code
        self.resample_rule = resample_rule
        self.df = pd.DataFrame()

    def _load_single(self, filepath, code):
        """内部方法：加载单品种数据"""
        print(f"[{code}] 正在加载数据: {filepath} ...")
        try:
            raw = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"Error: 文件未找到 {filepath}")
            return None

        # 1. 时间处理
        if 'tx_local_time' in raw.columns:
            raw['datetime'] = pd.to_datetime(raw['tx_local_time'], unit='ms')
            if raw['datetime'].dt.tz is None:
                raw['datetime'] = raw['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
            df = raw.set_index('datetime').sort_index()
        else:
            df = raw

        # 2. 重采样规则
        agg_rules = {
            'price': 'last', 'tick_vol': 'sum', 'tick_vwap': 'mean',
            'premium_rate': 'last', 'sentiment': 'last',
            'bp1':'last', 'bv1':'last', 'sp1':'last', 'sv1':'last'
        }
        # 兼容期货列
        if 'fut_price' in df.columns:
            agg_rules.update({'fut_price': 'last', 'fut_imb': 'mean'})
        
        # 兼容指数列
        if 'index_price' in df.columns:
            agg_rules['index_price'] = 'last'

        # 执行重采样
        df = df.resample(self.resample_rule).agg(agg_rules).ffill().dropna()
        
        # 3. 基础计算
        df['mid_price'] = (df['bp1'] + df['sp1']) / 2
        df['log_ret'] = np.log(df['mid_price'] / df['mid_price'].shift(1)).fillna(0)
        
        return df

    def load_and_merge(self):
        """加载双品种并对齐"""
        df_main = self._load_single(self.main_file, self.main_code)
        df_aux = self._load_single(self.aux_file, self.aux_code)
        
        if df_main is None: return None
        
        if df_aux is not None:
            # 重命名辅助标的列
            df_aux = df_aux.add_prefix('ctx_')
            # 必须 Inner Join 以对齐时间
            self.df = df_main.join(df_aux, how='inner')
            print(f"双品种对齐完成，共 {len(self.df)} 条样本")
        else:
            print("⚠️ 警告: 辅助标的数据缺失，仅使用单品种模式")
            self.df = df_main
            
        return self

    # ==============================================================================
    # 1. 经典微观因子 (L1 Microstructure) - 保持原味
    # ==============================================================================
    def calc_micro_factors(self):
        df = self.df
        
        # A. Smart VOI (资金流)
        db = df['bp1'].diff()
        ds = df['sp1'].diff()
        dvb = df['bv1'].diff()
        dvs = df['sv1'].diff()
        
        delta_vb = np.select([db > 0, db < 0], [df['bv1'], 0], default=dvb)
        delta_va = np.select([ds > 0, ds < 0], [0, df['sv1']], default=dvs)
        
        df['alpha_voi'] = delta_vb - delta_va
        vol_ma = df['tick_vol'].rolling(10).mean() + 1
        df['alpha_voi_smart'] = df['alpha_voi'].ewm(span=5).mean() / vol_ma

        # B. Micro-Price Dev (微观偏离)
        imb = df['bv1'] / (df['bv1'] + df['sv1'] + 1e-6)
        micro_price = df['bp1'] * (1 - imb) + df['sp1'] * imb
        df['alpha_micro_dev'] = (micro_price - df['mid_price']) / df['mid_price'] * 10000
        
        return self

    # ==============================================================================
    # 2. [新增] 跨品种博弈因子 (Cross-Asset Alphas)
    # ==============================================================================
    def calc_cross_factors(self):
        df = self.df
        
        # 必须确保有辅助数据
        if 'ctx_mid_price' not in df.columns:
            return self

        # --- F1: Relative Strength (RS, 相对强弱) ---
        # 逻辑: 恒指涨幅 - 恒科涨幅
        # 如果 RS > 0，说明资金在主拉恒指；如果 RS < 0，说明资金在拉恒科
        # 关键用法: 如果 RS 突然背离 (比如恒指没动，恒科暴涨)，恒指大概率会补涨
        df['alpha_cross_rs'] = df['log_ret'] - df['ctx_log_ret']
        
        # --- F2: Cross Sentiment Gap (情绪剪刀差) ---
        # 逻辑: 恒指情绪 - 恒科情绪
        # 两个指数底层成分股虽然不同，但大情绪应该一致。如果出现剪刀差，必有回归。
        df['alpha_cross_sent_diff'] = df['sentiment'] - df['ctx_sentiment']
        
        # --- F3: Cross Lead-Lag (互为领跑) ---
        # 逻辑: 谁先动？计算 5秒前的恒科涨幅 与 当前恒指涨幅 的差
        # 如果 5秒前恒科大涨，而现在恒指还没动，这就是机会
        lag_ret = df['ctx_log_ret'].shift(2) # 滞后2个tick (6秒)
        df['alpha_cross_lead'] = lag_ret - df['log_ret']
        
        # --- F4: Statistical Arb Spread (统计套利价差) ---
        # 逻辑: Log(Price_A) - Beta * Log(Price_B)
        # 简单起见，假设 Beta=1 (强相关)，计算价差 Z-Score
        # 这是一个强均值回归因子
        spread = np.log(df['mid_price']) - np.log(df['ctx_mid_price'])
        # 动态 Z-Score (过去 5分钟 = 100 ticks)
        spread_mean = spread.rolling(100).mean()
        spread_std = spread.rolling(100).std()
        df['alpha_cross_arb_z'] = (spread - spread_mean) / (spread_std + 1e-6)
        
        return self

    # ==============================================================================
    # 3. 期货与场景逻辑 (原有逻辑保留)
    # ==============================================================================
    def calc_scenario_factors(self):
        df = self.df
        minutes = df.index.hour * 60 + df.index.minute
        
        # A. Futures Lead (期货领跑)
        if 'fut_price' in df.columns:
            fut_ret = df['fut_price'].pct_change()
            df['alpha_fut_lead'] = fut_ret - df['log_ret'] # 差值越大，牵引力越大

        # B. Noon Arb (午盘套利)
        df['logic_noon_arb'] = 0.0
        mask_noon = (minutes >= 780) & (minutes <= 785) # 13:00-13:05
        
        if mask_noon.any() and 'index_price' in df.columns:
            # 指数收益 - ETF收益
            idx_ret = df['index_price'].pct_change()
            etf_ret = df['log_ret']
            df.loc[mask_noon, 'logic_noon_arb'] = (idx_ret - etf_ret) * 100 # 放大信号

        # C. Force Exit (尾盘清仓)
        df['logic_force_exit'] = 0.0
        mask_late = minutes >= 890 # 14:50
        time_left = 897 - minutes[mask_late]
        exit_signal = -20.0 / (time_left + 1.0)
        df.loc[mask_late, 'logic_force_exit'] = exit_signal
        
        return self

    def get_final_factors(self):
        self.load_and_merge()
        if self.df.empty: return pd.DataFrame()
        
        self.calc_micro_factors()
        self.calc_cross_factors()
        self.calc_scenario_factors()
        
        self.df = self.df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 筛选因子列
        cols = ['mid_price', 'bp1', 'sp1'] + \
               [c for c in self.df.columns if c.startswith('alpha_') or c.startswith('logic_')]
        
        return self.df[cols]

# ==============================================================================
# Main 测试
# ==============================================================================
if __name__ == "__main__":
    ff = FactorFactoryPro('sz159920.csv', 'sh513130.csv')
    df = ff.get_final_factors()
    
    if not df.empty:
        print("\n因子计算完成，Top 5 行:")
        print(df[['alpha_fut_lead', 'alpha_cross_rs', 'alpha_cross_arb_z']].tail())
        
        # 简单 IC 测试
        df['next_ret'] = np.log(df['mid_price'].shift(-5) / df['mid_price'])
        print("\n=== 新因子 IC 测试 (预测未来 15s) ===")
        print(f"Alpha Cross RS (相对强弱): {df['alpha_cross_rs'].corr(df['next_ret']):.4f}")
        print(f"Alpha Cross Arb Z (统计套利): {df['alpha_cross_arb_z'].corr(df['next_ret']):.4f}")