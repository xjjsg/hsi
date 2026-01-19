"""
HSI HFT V3 - Regime分析v1.1

基于用户v1.1诊断方案：
1. 字段口径统一
2. 分位数评分制
3. Action两阶段gating
4. 分layer的min_residence

验证目标：
1. Micro不塌缩（illiquid可进可出）
2. Action驻留长度进入可交易范围（>15 bars）
3. 特征健康度全程监控
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.getcwd())

from hsi_hft_v3.data_layer import V5DataLoader
from hsi_hft_v3.features.whitebox import WhiteBoxFeatureFactory
from hsi_hft_v3.regime_detector_v11 import (
    IntradayQuantileBaseline,
    TwoTierRegimeDetector_v11,
    CanonicalFeatureMapper,
    FeatureHealthMonitor,
)


def main():
    print("=" * 80)
    print("HSI Regime分析 v1.1 - 工程化验证")
    print("=" * 80)
    print()

    # ==========================================
    # Step 1: 加载数据
    # ==========================================
    print("[1] 加载数据...")
    loader = V5DataLoader(data_dir="./data")

    # 加载所有数据
    all_samples = []
    for day_samples in loader.iter_days():
        all_samples.extend(day_samples)

    print(f"✅ 加载 {len(all_samples)} 个样本\n")

    # ==========================================
    # Step 2: 建立Baseline（两遍扫描）
    # ==========================================
    print("[2] 建立日内分位数基线（Session分离）...")
    baseline = IntradayQuantileBaseline(bucket_minutes=5)
    wb_factory = WhiteBoxFeatureFactory()
    mapper = CanonicalFeatureMapper()

    # 第一遍：收集数据
    for i, s in enumerate(all_samples):
        wb_out = wb_factory.compute(s.bars)

        # 字段统一提取
        vpin = mapper.get_canonical_value(wb_out["white_target_raw"], "vpin")
        spread = mapper.get_canonical_value(wb_out["white_target_raw"], "spread_bps")
        depth = s.bars[-1].total_depth if s.bars else 10000.0

        if vpin is not None and spread is not None:
            baseline.add_observation(
                timestamp_ms=s.bars[-1].tx_local_time,
                vpin=abs(vpin),
                spread=spread,
                depth=depth,
            )

        if (i + 1) % 10000 == 0:
            print(f"  处理 {i+1}/{len(all_samples)}...")

    # 计算分位数
    baseline.compute_quantiles()
    print(f"✅ 建立基线，{len(baseline.quantile_table)} 个时间桶\n")

    # 🔍 输出baseline样例（诊断用）
    print("[DEBUG] Baseline样例（前5个bucket）:")
    for key in list(baseline.quantile_table.keys())[:5]:
        metrics = baseline.quantile_table[key]
        print(f"  {key}:")
        print(f"    spread p95: {metrics['spread'].get('p95', 0):.2f} bps")
        print(f"    depth p50: {metrics['depth'].get('p50', 0):.0f}")
        print(f"    vpin p90: {metrics['vpin'].get('p90', 0):.4f}")
    print()

    # ==========================================
    # Step 3: Regime检测
    # ==========================================
    print("[3] 运行两层Regime检测（v1.1）...")
    detector = TwoTierRegimeDetector_v11(
        baseline=baseline, min_residence_micro=10, min_residence_action=15
    )

    results = []
    feature_health_log = []

    for i, s in enumerate(all_samples):
        wb_out = wb_factory.compute(s.bars)

        # 提取canonical值
        vpin = mapper.get_canonical_value(wb_out["white_target_raw"], "vpin") or 0.0
        spread = (
            mapper.get_canonical_value(wb_out["white_target_raw"], "spread_bps") or 0.0
        )
        depth = s.bars[-1].total_depth if s.bars else 10000.0

        white_risk = {"vpin": abs(vpin), "spread_bps": spread, "depth": depth}

        # 检测
        mid = s.bars[-1].mid if s.bars else 0.0
        prev_mid = s.bars[-2].mid if len(s.bars) > 1 else mid

        micro, action, conf = detector.detect(
            timestamp_ms=s.bars[-1].tx_local_time,
            white_risk=white_risk,
            mid=mid,
            prev_mid=prev_mid,
        )

        # 记录
        results.append(
            {
                "timestamp": pd.Timestamp(s.bars[-1].tx_local_time, unit="ms"),
                "micro": micro,
                "action": action,
                "regime": f"{micro}:{action}",
                "confidence": conf,
                "vpin": abs(vpin),
                "spread": spread,
                "depth": depth,
            }
        )

        # 健康度日志（每1000个样本）
        if (i + 1) % 1000 == 0:
            vpin_health, vpin_reason = detector.health_monitor.is_healthy("vpin")
            spread_health, spread_reason = detector.health_monitor.is_healthy("spread")
            depth_health, depth_reason = detector.health_monitor.is_healthy("depth")

            feature_health_log.append(
                {
                    "sample": i + 1,
                    "vpin_health": vpin_health,
                    "vpin_reason": vpin_reason,
                    "spread_health": spread_health,
                    "spread_reason": spread_reason,
                    "depth_health": depth_health,
                    "depth_reason": depth_reason,
                }
            )

        if (i + 1) % 10000 == 0:
            print(f"  处理 {i+1}/{len(all_samples)}...")

    print(f"✅ 检测完成，共 {len(results)} bars\n")

    # ==========================================
    # Step 4: 统计分析
    # ==========================================
    print("[4] 统计分析")
    print("-" * 80)

    df = pd.DataFrame(results)

    # Regime分布
    print("\n【Regime分布】")
    regime_counts = df["regime"].value_counts()
    print(regime_counts)

    print("\n占比（%）：")
    regime_pct = (regime_counts / len(df) * 100).round(2)
    print(regime_pct)

    # Micro层单独统计
    print("\n【Micro层分布】")
    micro_counts = df["micro"].value_counts()
    micro_pct = (micro_counts / len(df) * 100).round(2)
    for micro, pct in micro_pct.items():
        print(f"  {micro}: {pct}%")

    # 切换分析
    print("\n【切换频率】")
    regime_changes = (df["regime"] != df["regime"].shift()).sum() - 1
    print(f"Regime总切换次数: {regime_changes}")
    print(f"平均切换间隔: {len(df) / (regime_changes + 1):.1f} bars")

    micro_changes = (df["micro"] != df["micro"].shift()).sum() - 1
    action_changes = (df["action"] != df["action"].shift()).sum() - 1
    print(f"Micro切换: {micro_changes} 次")
    print(f"Action切换: {action_changes} 次")

    # 驻留时间
    print("\n【驻留时间统计】")
    regime_runs = []
    current_regime = None
    run_length = 0

    for regime in df["regime"]:
        if regime == current_regime:
            run_length += 1
        else:
            if current_regime is not None:
                regime_runs.append({"regime": current_regime, "length": run_length})
            current_regime = regime
            run_length = 1

    if current_regime is not None:
        regime_runs.append({"regime": current_regime, "length": run_length})

    runs_df = pd.DataFrame(regime_runs)
    residence_stats = runs_df.groupby("regime")["length"].describe()
    print(residence_stats)

    # 🔍 特征健康度报告
    print("\n【特征健康度】")
    health_df = pd.DataFrame(feature_health_log)
    if len(health_df) > 0:
        print(f"VPIN健康率: {health_df['vpin_health'].mean()*100:.1f}%")
        print(f"Spread健康率: {health_df['spread_health'].mean()*100:.1f}%")
        print(f"Depth健康率: {health_df['depth_health'].mean()*100:.1f}%")

        # 不健康原因统计
        if not health_df["vpin_health"].all():
            print(f"\nVPIN不健康原因:")
            print(health_df[~health_df["vpin_health"]]["vpin_reason"].value_counts())

    # 🔍 Illiquid锁死检查
    print("\n【Illiquid锁死检查】")
    illiquid_runs = runs_df[runs_df["regime"].str.startswith("illiquid")]
    if len(illiquid_runs) > 0:
        max_illiquid_run = illiquid_runs["length"].max()
        total_illiquid = illiquid_runs["length"].sum()
        print(f"最长illiquid驻留: {max_illiquid_run} bars")
        print(f"Illiquid段数: {len(illiquid_runs)}")
        print(f"平均illiquid长度: {illiquid_runs['length'].mean():.1f} bars")

        if max_illiquid_run > len(df) * 0.9:
            print("⚠️ WARNING: Illiquid出现超长驻留（>90%数据），可能锁死！")
        else:
            print("✅ Illiquid可正常进出")

    # ==========================================
    # Step 5: 保存结果
    # ==========================================
    print("\n[5] 保存结果...")
    os.makedirs("./analysis/regime_analysis_v11", exist_ok=True)

    df.to_csv("./analysis/regime_analysis_v11/regime_history.csv", index=False)
    residence_stats.to_csv("./analysis/regime_analysis_v11/residence_stats.csv")

    if len(health_df) > 0:
        health_df.to_csv(
            "./analysis/regime_analysis_v11/feature_health.csv", index=False
        )

    print("✅ 结果保存到 ./analysis/regime_analysis_v11/\n")

    print("=" * 80)
    print("分析完成！")
    print("=" * 80)

    print("\n【v1.1改进验证】")
    print("1. 检查Micro层是否塌缩 -> 见【Illiquid锁死检查】")
    print("2. 检查Action驻留长度 -> 见【驻留时间统计】（目标>15 bars）")
    print("3. 检查特征健康度 -> 见【特征健康度】（目标>80%）")
    print("4. 检查切换频率 -> 见【切换频率】（目标<10次/小时）")


if __name__ == "__main__":
    main()
