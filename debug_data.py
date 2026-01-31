import os
import pandas as pd
import numpy as np
import glob

DATA_DIR = "./data"


def check_file(filepath):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"❌ 读取错误 {filepath}: {e}")
        return False

    # 检查列
    # 假设标准列: price, tick_vol, tick_amt
    # 同时也检查 'mid'

    issues = []

    # 1. NaN/Inf 检查
    if df.isnull().values.any():
        issues.append(f"包含 {df.isnull().sum().sum()} 个 NaN")

    # 检查数值列是否包含无穷大
    num_cols = df.select_dtypes(include=[np.number]).columns
    if np.isinf(df[num_cols].values).any():
        issues.append("包含无穷大 (Infinity) 值")

    # 2. 零值检查 (价格不应为0)
    if "price" in df.columns:
        if (df["price"] <= 0).any():
            issues.append(f"包含 {(df['price']<=0).sum()} 个 零/负 价格")

    # 3. 极端值检查 (3秒内对数收益率跳升 > 10% 对于ETF来说可疑)
    # 简单的价格跳变检查
    if "price" in df.columns:
        pct_change = df["price"].pct_change().abs()
        max_chg = pct_change.max()
        if max_chg > 0.10:  # > 10% jump between ticks
            issues.append(f"极端价格跳变: {max_chg:.2%}")

    if issues:
        print(f"⚠️ {os.path.basename(filepath)}: {', '.join(issues)}")
        return False
    else:
        return True


def main():
    print("🔍 扫描数据异常...")
    files = glob.glob(f"{DATA_DIR}/**/*.csv", recursive=True)
    files = sorted(files)

    valid_count = 0
    issue_count = 0

    for f in files:
        if "summary" in f:
            continue
        if check_file(f):
            valid_count += 1
        else:
            issue_count += 1

    print(f"\n扫描完成. 有效: {valid_count}, 异常: {issue_count}")


if __name__ == "__main__":
    main()
