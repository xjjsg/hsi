import os
import pandas as pd
import numpy as np
import glob

DATA_DIR = "./data"


def check_file(filepath):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return False

    # Check Columns
    # Assuming standard columns: price, tick_vol, tick_amt
    # Also check 'mid' if it exists or constructed

    issues = []

    # 1. NaN/Inf Check
    if df.isnull().values.any():
        issues.append(f"Contains {df.isnull().sum().sum()} NaNs")

    # Check numeric columns for Inf
    num_cols = df.select_dtypes(include=[np.number]).columns
    if np.isinf(df[num_cols].values).any():
        issues.append("Contains Infinity values")

    # 2. Zero Check (Price shouldn't be 0)
    if "price" in df.columns:
        if (df["price"] <= 0).any():
            issues.append(f"Contains {(df['price']<=0).sum()} Zero/Negative Prices")

    # 3. Extreme Value Check (Log Return Spike > 10% in 3s is suspicious for ETF)
    # Simple check on price jump
    if "price" in df.columns:
        pct_change = df["price"].pct_change().abs()
        max_chg = pct_change.max()
        if max_chg > 0.10:  # > 10% jump between ticks
            issues.append(f"Extreme Price Jump: {max_chg:.2%}")

    if issues:
        print(f"⚠️ {os.path.basename(filepath)}: {', '.join(issues)}")
        return False
    else:
        return True


def main():
    print("🔍 Scanning Data for Anomalies...")
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

    print(f"\nScan Complete. Valid: {valid_count}, Issues: {issue_count}")


if __name__ == "__main__":
    main()
