#!/usr/bin/env python3
import pandas as pd
import sys
import os

def to_markdown(df: pd.DataFrame) -> str:
    """
    Try pandas' to_markdown; if tabulate is missing, fall back to manual Markdown.
    """
    try:
        return df.to_markdown(index=False)
    except ImportError:
        cols = list(df.columns)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = []
        for _, row in df.iterrows():
            cells = [str(row[c]) for c in cols]
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join([header, sep] + rows)

def format_default(x):
    """
    For general float columns: if integer, show as int; otherwise pad to 4 decimal places.
    """
    try:
        v = float(x)
    except:
        return x
    if v.is_integer():
        return str(int(v))
    return f"{v:.4f}"

def format_hr(x):
    """
    For HR columns: if integer, show as int; otherwise pad to 3 decimal places.
    """
    try:
        v = float(x)
    except:
        return x
    if v.is_integer():
        return str(int(v))
    return f"{v:.3f}"

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply formatting and highlight best results:
      - Columns ending with '↑': highlight max (bold).
      - Columns ending with '↓': highlight min (bold).
      - HR columns use 3 decimals; others use 4 decimals.
    """
    # Apply formatting
    for col in df.columns:
        if df[col].dtype.kind in 'f':
            if col.startswith("HR@"):
                df[col] = df[col].apply(format_hr)
            else:
                df[col] = df[col].apply(format_default)

    # Highlight best
    for col in df.columns:
        if col in ("SampleMethod", "Model"):
            continue
        arrow_up = col.endswith("↑")
        arrow_down = col.endswith("↓")
        if not (arrow_up or arrow_down):
            # default: assume upward is better
            arrow_up = True
        # convert values back to float for comparison
        nums = []
        for v in df[col]:
            try:
                nums.append(float(v))
            except:
                nums.append(None)
        valid = [(i, n) for i, n in enumerate(nums) if n is not None]
        if not valid:
            continue
        best_idx = (
            max(valid, key=lambda x: x[1])[0] if arrow_up else min(valid, key=lambda x: x[1])[0]
        )
        cell = df.iat[best_idx, df.columns.get_loc(col)]
        df.iat[best_idx, df.columns.get_loc(col)] = f"<mark>**{cell}**</mark>"
    return df

def main():
    # Accept CSV path or default
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "/scratch/user/chuanhsin0110/ClusterExposure-DPO/experiments/metrics/metrics_summary.csv"
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Ensure headers have arrows
    def add_arrow(c):
        if "Model" == c:
            return c
        if "NDCG@" in c and "↑" not in c:
            return c + " ↑"
        if "HR@" in c and "↑" not in c:
            return c + " ↑"
        if "Diversity" in c and "↑" not in c:
            return c + " ↑"
        if "DivRatio" in c and "↑" not in c:
            return c + " ↑"
        if "DGU" in c and "↓" not in c:
            return c + " ↓"
        if "MGU" in c and "↓" not in c:
            return c + " ↓"
        if "ORRatio" in c and "↓" not in c:
            return c + " ↓"
        if "Predict_NotIn_Ratio" in c and "↓" not in c:
            return c + " ↓"
        return c

    df.columns = [add_arrow(c) for c in df.columns]

    # Columns to always include
    always = ["Model", "SampleMethod"]

    # Top-5 Metrics
    cols5 = [
        c for c in df.columns
        if ("@5" in c and "PredictNotInRatio" not in c) or c in always
    ]
    df5 = df[cols5].copy()
    df5 = prepare_df(df5)
    print("### Top-5 Metrics\n")
    print(to_markdown(df5))
    print("\n")

    # Top-10 Metrics
    cols10 = [
        c for c in df.columns
        if ("@10" in c and "PredictNotInRatio" not in c) or c in always
    ]
    df10 = df[cols10].copy()
    df10 = prepare_df(df10)
    print("### Top-10 Metrics\n")
    print(to_markdown(df10))
    print("\n")

    # Predict Not-In-Ratio
    col_pred = "Predict_NotIn_Ratio ↓"
    if col_pred in df.columns:
        dfp = df[[*always, col_pred]].copy()
        dfp = prepare_df(dfp)
        print("### Predict Not-In-Ratio\n")
        print(to_markdown(dfp))
    else:
        print(f"(Column `{col_pred}` not found in CSV)")

if __name__ == "__main__":
    main()
