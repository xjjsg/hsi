import pandas as pd
import numpy as np

# ==============================================================================
# 核心类: 因子工厂
# ==============================================================================
class FactorFactory:
    def __init__(self, filepath, code, resample_rule='3s'):
        self.filepath = filepath
        self.code = code
        self.resample_rule = resample_rule
        self.raw_df = pd.DataFrame()
        self.df = pd.DataFrame()

    def load_and_clean(self):
        """加载并清洗数据 (3秒聚合)"""
        print(f"[{self.code}] 正在加载数据: {self.filepath} ...")
        try:
            self.raw_df = pd.read_csv(self.filepath)
        except FileNotFoundError:
            print(f"Error: 文件未找到 {self.filepath}")
            return self

        # 1. 时间戳解析
        self.raw_df['datetime'] = pd.to_datetime(self.raw_df['tx_local_time'], unit='ms')
        if self.raw_df['datetime'].dt.tz is None:
            self.raw_df['datetime'] = self.raw_df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
        
        self.raw_df = self.raw_df.set_index('datetime').sort_index()

        # 2. 重采样规则 (聚合逻辑)
        agg_rules = {
            'price': 'last', 'tick_vol': 'sum', 'tick_amt': 'sum', 'tick_vwap': 'mean',
            'premium_rate': 'last', 'iopv': 'last', 'sentiment': 'last',
            'bp1': 'last', 'bv1': 'last', 'sp1': 'last', 'sv1': 'last',
        }
        
        # 兼容可选列
        if 'index_price' in self.raw_df.columns: agg_rules['index_price'] = 'last'
        if 'fut_price' in self.raw_df.columns:
            agg_rules.update({'fut_price': 'last', 'fut_imb': 'mean', 'fx_rate': 'last'})

        # 3. 执行重采样 (ffill处理空洞，但dropna去除无行情时段)
        self.df = self.raw_df.resample(self.resample_rule).agg(agg_rules).ffill().dropna()

        # 4. 基础衍生列
        self.df['mid_price'] = (self.df['bp1'] + self.df['sp1']) / 2
        self.df = self.df[self.df['mid_price'] > 0] # 剔除异常

        print(f"[{self.code}] 清洗完成，有效样本数: {len(self.df)}")
        return self

    def calc_micro_factors(self):
        """计算微观结构因子 (L1 Data)"""
        df = self.df
        
        # --- F1: Smart VOI (资金流) ---
        # 逻辑: 即使只有一档，也能通过价格变动判断成交是主动买还是主动卖
        db = df['bp1'].diff()
        ds = df['sp1'].diff()
        dvb = df['bv1'].diff()
        dvs = df['sv1'].diff()

        # Bid侧增量
        delta_vb = np.select([db > 0, db < 0], [df['bv1'], 0], default=dvb)
        # Ask侧增量
        delta_va = np.select([ds > 0, ds < 0], [0, df['sv1']], default=dvs)

        df['alpha_voi'] = delta_vb - delta_va
        # 平滑处理: 除以近期成交量均值，防止开盘放量时的信号漂移
        vol_ma = df['tick_vol'].rolling(10).mean() + 1
        df['alpha_voi_smart'] = df['alpha_voi'].ewm(span=5).mean() / vol_ma

        # --- F2: Micro-Price Dev (微观价格偏离) ---
        # 逻辑: 挂单失衡修正后的中间价。bv1大则价格倾向涨。
        imb = df['bv1'] / (df['bv1'] + df['sv1'] + 1e-6)
        df['micro_price'] = df['bp1'] * (1 - imb) + df['sp1'] * imb
        # 因子: MicroPrice 相对于 MidPrice 的万分比偏离
        df['alpha_micro_dev'] = (df['micro_price'] - df['mid_price']) / df['mid_price'] * 10000

        # --- F3: Spread Pressure (价差压力) ---
        # 逻辑: 价差越大，流动性越差，也是成本越高的体现
        df['alpha_spread_bps'] = (df['sp1'] - df['bp1']) / df['mid_price'] * 10000
        
        return self

    def calc_cross_asset_factors(self):
        """计算跨品种博弈因子"""
        df = self.df

        # --- F4: Futures Lead (期货领跑) ---
        # 最强因子: 期货涨幅 - 现货涨幅
        if 'fut_price' in df.columns:
            # 计算 3秒(1 tick) 变化差
            fut_ret = df['fut_price'].pct_change()
            etf_ret = df['price'].pct_change()
            df['alpha_fut_lead'] = fut_ret - etf_ret
            # 期货盘口失衡 (新浪数据自带)
            df['alpha_fut_imb'] = df['fut_imb']

        # --- F5: Index Lead (指数领跑) ---
        if 'index_price' in df.columns:
            idx_ret = df['index_price'].pct_change()
            etf_ret = df['price'].pct_change()
            df['alpha_idx_lead'] = idx_ret - etf_ret

        # --- F6: Sentiment Divergence (情绪背离) ---
        # 过去30秒(10个tick)的情绪变化 vs 价格变化
        sent_chg = df['sentiment'].diff(10)
        price_chg = df['price'].pct_change(10) * 100
        df['alpha_sent_divergence'] = sent_chg - price_chg
        
        # --- F7: Premium Z-Score (折溢价回归) ---
        mean_prem = df['premium_rate'].rolling(100).mean()
        std_prem = df['premium_rate'].rolling(100).std()
        df['alpha_premium_z'] = (df['premium_rate'] - mean_prem) / (std_prem + 1e-6)

        return self

    def calc_scenario_logic(self):
        """计算时间场景逻辑 (开盘过滤、午盘套利、尾盘清仓)"""
        df = self.df
        minutes = df.index.hour * 60 + df.index.minute
        
        # --- S1: Filter Instability (开盘波动过滤) ---
        # 目的: 09:30-09:35 寻找合适入场点，避开巨大价差
        # 逻辑: 如果价差 > 15bp (0.15%)，标记为不稳定，建议 WAIT
        df['filter_unstable'] = 0.0
        mask_wide_spread = df['alpha_spread_bps'] > 15 
        # 仅在开盘前10分钟生效
        mask_open = minutes < (9*60 + 40)
        df.loc[mask_wide_spread & mask_open, 'filter_unstable'] = 1.0

        # --- S2: Noon Gap Arb (午盘套利) ---
        # 目的: 捕捉 13:00 开盘瞬间，指数在午休期间的涨跌幅传导
        df['logic_noon_arb'] = 0.0
        # 13:00 - 13:05
        mask_noon = (minutes >= 780) & (minutes <= 785)
        
        if mask_noon.any() and 'alpha_idx_lead' in df.columns:
            # 此时指数领跑因子如果很大，大概率是真实的补涨/补跌需求
            # 放大权重 5 倍
            df.loc[mask_noon, 'logic_noon_arb'] = df.loc[mask_noon, 'alpha_idx_lead'] * 5.0

        # --- S3: Force Exit (尾盘强制清仓) ---
        # 目的: 14:50 后生成强力卖出信号，确保日内平仓
        df['logic_force_exit'] = 0.0
        mask_late = minutes >= 890 # 14:50
        
        # 信号随时间指数级增强 (负值)
        # 14:50 -> -2, 14:56 -> -20
        time_left = 897 - minutes[mask_late]
        exit_signal = -20.0 / (time_left + 1.0)
        
        df.loc[mask_late, 'logic_force_exit'] = exit_signal

        return self

    def get_final_factors(self):
        """生成最终因子表"""
        self.load_and_clean()
        if self.df.empty: return pd.DataFrame()
        
        self.calc_micro_factors()
        self.calc_cross_asset_factors()
        self.calc_scenario_logic()
        
        self.df = self.df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 导出列：价格 + Alpha因子 + 逻辑因子 + 过滤因子
        cols = ['price', 'mid_price', 'bp1', 'sp1'] + \
               [c for c in self.df.columns if c.startswith('alpha_') or c.startswith('logic_') or c.startswith('filter_')]
        
        return self.df[cols]

# ==============================================================================
# 辅助类: 因子评估器 (包含实战成本模拟)
# ==============================================================================
class FactorEvaluator:
    def __init__(self, df):
        self.df = df.copy()

    def evaluate_ic(self, horizons=[1, 5]):
        """计算 IC (预测能力)"""
        print("\n=== IC (预测能力) 评估 ===")
        report = []
        
        # 准备标签
        for h in horizons:
            self.df[f'ret_{h}'] = np.log(self.df['mid_price'].shift(-h) / self.df['mid_price'])
            
        factors = [c for c in self.df.columns if c.startswith('alpha_')]
        
        for f in factors:
            row = {'Factor': f}
            for h in horizons:
                # Spearman IC
                valid = self.df[[f, f'ret_{h}']].dropna()
                ic = valid[f].corr(valid[f'ret_{h}'], method='spearman')
                row[f'IC_{h*3}s'] = round(ic, 4)
            report.append(row)
            
        res_df = pd.DataFrame(report).set_index('Factor')
        res_df = res_df.sort_values(by=res_df.columns[0], key=abs, ascending=False)
        print(res_df)
        return res_df

    def evaluate_real_cost_pnl(self, signal_col, hold_period=5):
        """
        [关键] 实战盈亏模拟
        模拟配置: 万一佣金 (0.0001), 1跳点差 (10bp), Taker吃单模式
        """
        commission = 0.0001  # 万一
        
        print(f"\n⚡ 实战模拟 [{signal_col}] (佣金:万1, 持仓:{hold_period*3}秒)")
        
        # 1. 信号生成 (Top 5% 强信号)
        threshold = self.df[signal_col].quantile(0.95)
        long_signals = self.df[signal_col] > threshold
        
        if not long_signals.any():
            print("无触发信号")
            return

        sim_data = self.df.loc[long_signals].copy()
        
        # 2. 交易逻辑 (Taker)
        # 买入: 吃卖一价 (sp1)
        sim_data['entry_price'] = sim_data['sp1']
        # 卖出: N秒后的买一价 (bp1)
        sim_data['exit_price'] = self.df['bp1'].shift(-hold_period).loc[sim_data.index]
        
        # 3. 盈亏计算
        # 毛利
        sim_data['gross_ret'] = (sim_data['exit_price'] - sim_data['entry_price']) / sim_data['entry_price']
        # 净利 (扣除双边佣金)
        sim_data['net_ret'] = sim_data['gross_ret'] - (commission * 2)
        
        # 4. 结果统计
        valid = sim_data.dropna()
        if len(valid) == 0: return
        
        avg_net_bp = valid['net_ret'].mean() * 10000
        win_rate = (valid['net_ret'] > 0).mean()
        
        # 成本拆解
        avg_price = valid['entry_price'].mean()
        spread_cost_bp = (0.001 / avg_price) * 10000 # 假设tick_size=0.001
        
        print(f"  信号阈值: > {threshold:.4f}")
        print(f"  交易次数: {len(valid)}")
        print(f"  点差成本: {spread_cost_bp:.2f} bp (最大敌人)")
        print(f"  佣金成本: 2.00 bp")
        print(f"  ---------------------------")
        print(f"  💰 平均净利: {avg_net_bp:.2f} bp")
        print(f"  🏆 胜率:     {win_rate:.2%}")
        
        if avg_net_bp > 0:
            print("  ✅ 策略可行 (Taker模式)")
        else:
            print("  ❌ 策略亏损 (建议改用 Maker 挂单模式)")


# ==============================================================================
# Main 执行入口 (测试用)
# ==============================================================================
if __name__ == "__main__":
    # 1. 构建因子
    factory = FactorFactory(filepath='sz159920.csv', code='159920')
    df = factory.get_final_factors()
    
    if not df.empty:
        print("\n因子构建完成，样本预览:")
        print(df[['mid_price', 'alpha_fut_lead', 'logic_force_exit']].tail())
        
        # 2. 评估因子
        evaluator = FactorEvaluator(df)
        
        # (A) 看预测能力
        evaluator.evaluate_ic(horizons=[1, 5]) # 3s, 15s