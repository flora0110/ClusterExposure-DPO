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
        # Manual fallback
        cols = list(df.columns)
        # header
        header = "| " + " | ".join(cols) + " |"
        # separator
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        # rows
        rows = []
        for _, row in df.iterrows():
            cells = [str(row[c]) for c in cols]
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join([header, sep] + rows)

def main():
    # 你可以通过命令行参数传入 CSV 路径
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "/scratch/user/chuanhsin0110/ClusterExposure-DPO/experiments/metrics/metrics_summary.csv"

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Top-5
    cols5 = [c for c in ["SampleMethod","NDCG@5","HR@5","Diversity@5","DivRatio@5","DGU@5","MGU@5","ORRatio@5"] if c in df.columns]
    df5 = df[cols5]
    print("### Top-5 Metrics\n")
    print(to_markdown(df5))
    print("\n")

    # Top-10
    cols10 = [c for c in ["SampleMethod","NDCG@10","HR@10","Diversity@10","DivRatio@10","DGU@10","MGU@10","ORRatio@10"] if c in df.columns]
    df10 = df[cols10]
    print("### Top-10 Metrics\n")
    print(to_markdown(df10))
    print("\n")

    # Predict Not-In-Ratio
    col_pred = "Predict_NotIn_Ratio"
    if col_pred in df.columns:
        dfp = df[["SampleMethod", col_pred]]
        print("### Predict Not-In-Ratio\n")
        print(to_markdown(dfp))
    else:
        print(f"(Column `{col_pred}` not found in CSV)")

if __name__ == "__main__":
    main()
