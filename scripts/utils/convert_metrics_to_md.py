#!/usr/bin/env python3
import pandas as pd
import sys
import os
import numpy as np

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

def format_cell(val):
    """
    Round floats to 4 decimal places unless they are integer-valued.
    Leave integers and non-numerics unchanged.
    """
    if isinstance(val, float):
        if np.isclose(val, round(val)):
            # treat as integer
            return int(round(val))
        else:
            return round(val, 4)
    else:
        return val

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply format_cell to every cell in the DataFrame.
    """
    return df.applymap(format_cell)

def main():
    # accept csv path as argument or default
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "/scratch/user/chuanhsin0110/ClusterExposure-DPO/experiments/metrics/metrics_summary.csv"

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Top-5
    cols5 = [c for c in ["Model","NDCG@5","HR@5","Diversity@5","DivRatio@5","DGU@5","MGU@5","ORRatio@5"] if c in df.columns]
    df5 = df[cols5]
    df5 = prepare_df(df5)
    print("### Top-5 Metrics\n")
    print(to_markdown(df5))
    print("\n")

    # Top-10
    cols10 = [c for c in ["Model","NDCG@10","HR@10","Diversity@10","DivRatio@10","DGU@10","MGU@10","ORRatio@10"] if c in df.columns]
    df10 = df[cols10]
    df10 = prepare_df(df10)
    print("### Top-10 Metrics\n")
    print(to_markdown(df10))
    print("\n")

    # Predict Not-In-Ratio
    col_pred = "Predict_NotIn_Ratio"
    if col_pred in df.columns:
        dfp = df[["Model", col_pred]]
        dfp = prepare_df(dfp)
        print("### Predict Not-In-Ratio\n")
        print(to_markdown(dfp))
    else:
        print(f"(Column `{col_pred}` not found in CSV)")

if __name__ == "__main__":
    main()

