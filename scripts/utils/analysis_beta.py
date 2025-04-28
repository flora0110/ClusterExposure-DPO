#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os

beta_min = 0.05
beta_max = 0.15
t = "1.0"

def load_betas(jsonl_path):
    """
    从 JSONL 文件读取 adapted_betas，
    返回一个扁平化的 float 列表。
    """
    betas = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            # entry 是一个 dict: { "rejected1": 0.123, ... }
            betas.extend(float(v) for v in entry.values())
    return betas

def summarize(betas):
    """
    打印统计量：均值、最小、最大、中位数、四分位数。
    """
    arr = np.array(betas)
    mean = arr.mean()
    mn   = arr.min()
    mx   = arr.max()
    median = np.median(arr)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)

    print("β 值统计摘要：")
    print(f"  平均值    (mean)    : {mean:.6f}")
    print(f"  最小值    (min)     : {mn:.6f}")
    print(f"  第一四分位 (Q1)     : {q1:.6f}")
    print(f"  中位数    (median)  : {median:.6f}")
    print(f"  第三四分位 (Q3)     : {q3:.6f}")
    print(f"  最大值    (max)     : {mx:.6f}")

def plot_histogram(betas, bins=30):
    """
    绘制 β 值的直方图。
    """
    plt.figure()
    plt.hist(betas, bins=bins)
    plt.title("Distribution of adapted β values")
    plt.xlabel("β value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_histogram_and_boxplot(betas, bins=30):
    """
    繪製直方圖 + 盒狀圖。
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # 直方圖
    axes[0].hist(betas, bins=bins)
    axes[0].set_title("Distribution of adapted β values (Histogram)")
    axes[0].set_xlabel("β value")
    axes[0].set_ylabel("Frequency")

    # 盒狀圖
    axes[1].boxplot(betas, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    axes[1].set_title("Boxplot of adapted β values")
    axes[1].set_xlabel("β value")

    plt.tight_layout()
    plt.show()

def plot_and_save(betas, save_dir):
    """畫直方圖+盒狀圖，並把圖存進指定資料夾。"""
    os.makedirs(save_dir, exist_ok=True)  # 自動建立目錄

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # 直方圖
    axes[0].hist(betas, bins=30)
    axes[0].set_title("Distribution of adapted β values (Histogram)")
    axes[0].set_xlabel("β value")
    axes[0].set_ylabel("Frequency")

    # 盒狀圖
    axes[1].boxplot(betas, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    axes[1].set_title("Boxplot of adapted β values")
    axes[1].set_xlabel("β value")

    plt.tight_layout()

    # 存檔
    histogram_path = os.path.join(save_dir, f"beta_histogramt_{t}__epoch_1_dpc_{beta_min}_{beta_max}.png")
    boxplot_path = os.path.join(save_dir, f"beta_boxplott_{t}__epoch_1_dpc_{beta_min}_{beta_max}.png")
    combined_path = os.path.join(save_dir, f"beta_histogram_boxplott_{t}__epoch_1_dpc_{beta_min}_{beta_max}.png")

    # 分開存
    fig.savefig(combined_path)
    print(f"✅ 圖片已儲存：{combined_path}")

    # 如果要單獨存 histogram、boxplot
    plt.figure()
    plt.hist(betas, bins=30)
    plt.title("Histogram of β values")
    plt.xlabel("β value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(histogram_path)
    print(f"✅ Histogram 圖儲存：{histogram_path}")

    plt.figure()
    plt.boxplot(betas, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title("Boxplot of β values")
    plt.xlabel("β value")
    plt.tight_layout()
    plt.savefig(boxplot_path)
    print(f"✅ Boxplot 圖儲存：{boxplot_path}")

def main():
    # 默认路径，可改成你自己的
    default_path = f"/scratch/user/chuanhsin0110/ClusterExposure-DPO/experiments/log/sigmoid/t_{t}_epoch_1_dpc_{beta_min}_{beta_max}/adapted_betas.jsonl"
    path = sys.argv[1] if len(sys.argv) > 1 else default_path

    if not Path(path).exists():
        print(f"ERROR: 找不到文件: {path}", file=sys.stderr)
        sys.exit(1)

    betas = load_betas(path)
    if not betas:
        print("No β values found in the log file.")
        sys.exit(0)
    

    summarize(betas)
    plot_and_save(betas, "/scratch/user/chuanhsin0110/ClusterExposure-DPO/experiments/log/sigmoid/")

if __name__ == "__main__":
    main()
