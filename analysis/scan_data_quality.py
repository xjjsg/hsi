"""
数据质量扫描脚本
检查所有数据文件的时间戳唯一性
"""

import pandas as pd
import os
import glob


def check_data_quality(base_dir="./data"):
    """扫描所有CSV文件检查时间戳异常"""

    print("=" * 80)
    print("数据质量扫描报告")
    print("=" * 80)

    symbols = ["sz159920", "sh513130"]
    issues = []
    total_files = 0

    for symbol in symbols:
        pattern = os.path.join(base_dir, symbol, "*.csv")
        files = sorted(glob.glob(pattern))

        print(f"\n【{symbol}】")
        print(f"总文件数: {len(files)}")

        for file in files:
            total_files += 1
            try:
                df = pd.read_csv(file)
                total_rows = len(df)

                if "tx_local_time" in df.columns:
                    unique_ts = df["tx_local_time"].nunique()
                    unique_ratio = unique_ts / max(total_rows, 1)

                    # 检测异常
                    status = "✅"
                    if unique_ratio < 0.01:
                        status = "🔴 CRITICAL"
                        issues.append(
                            {
                                "file": os.path.basename(file),
                                "total_rows": total_rows,
                                "unique_ts": unique_ts,
                                "ratio": unique_ratio,
                                "severity": "CRITICAL",
                            }
                        )
                    elif unique_ratio < 0.5:
                        status = "⚠️ WARNING"
                        issues.append(
                            {
                                "file": os.path.basename(file),
                                "total_rows": total_rows,
                                "unique_ts": unique_ts,
                                "ratio": unique_ratio,
                                "severity": "WARNING",
                            }
                        )
                    elif unique_ratio < 0.9:
                        status = "🟡 INFO"

                    if status != "✅":
                        print(
                            f"  {status} {os.path.basename(file)}: "
                            f"{total_rows}行 → {unique_ts}个唯一时间戳 "
                            f"({unique_ratio*100:.1f}%)"
                        )
                else:
                    print(f"  ⚠️ {os.path.basename(file)}: 缺少tx_local_time列")

            except Exception as e:
                print(f"  ❌ {os.path.basename(file)}: 读取失败 - {e}")

    # 汇总报告
    print("\n" + "=" * 80)
    print("扫描汇总")
    print("=" * 80)
    print(f"总扫描文件: {total_files}")
    print(f"发现问题: {len(issues)}")

    if issues:
        print(f"\n【问题详情】")
        critical = [i for i in issues if i["severity"] == "CRITICAL"]
        warning = [i for i in issues if i["severity"] == "WARNING"]

        if critical:
            print(f"\n🔴 严重问题 ({len(critical)}个):")
            for issue in critical:
                print(
                    f"  {issue['file']}: {issue['total_rows']}行 → "
                    f"{issue['unique_ts']}个时间戳 ({issue['ratio']*100:.2f}%)"
                )

        if warning:
            print(f"\n⚠️ 警告 ({len(warning)}个):")
            for issue in warning:
                print(
                    f"  {issue['file']}: {issue['total_rows']}行 → "
                    f"{issue['unique_ts']}个时间戳 ({issue['ratio']*100:.2f}%)"
                )

        print(f"\n建议: 删除有严重问题的文件")
        if critical:
            print("删除命令:")
            for issue in critical:
                for symbol in symbols:
                    print(f"  Remove-Item ./data/{symbol}/{issue['file']}")
    else:
        print("✅ 所有文件数据质量良好！")

    print("=" * 80)
    return issues


if __name__ == "__main__":
    check_data_quality()
