# ClusterExposure-DPO: LLM-based Recommendation via Cluster-Aware Exposure-Aware Negative Sampling

## Inspiration & Methodology

This project is primarily inspired by two papers:

- [SPRec](https://arxiv.org/abs/2412.09243): Self-Play to Debias LLM-based Recommendation

    ↳ Introduces self-play for negative sampling by using LLM-generated predictions as rejections.

- [S-DPO](https://arxiv.org/abs/2406.09215): Softmax Direct Preference Optimization (not yet been fully reproduced)

    ↳ Proposes the use of multiple negative samples in the DPO framework.

Building on SPRec's self-play foundation, this work proposes a Cluster-Aware and Exposure-Aware Negative Sampling framework, which selects hard negatives based on both:

- User preference clusters (inferred from past behavior)

- Exposure statistics (popularity and frequency of each item)

We combine these signals to generate more challenging and diverse negative samples, and evaluate their effectiveness under DPO fine-tuning across eight distinct sampling strategies.


## Pipeline

Our training pipeline consists of:

1. Self-Play Generation
    
    - Generate candidate responses via beam search with constrained decoding.

2. Candidate Annotation
    
    - Annotate each rejected candidate with genre and exposure metadata.

3. Negative Sampling
    
    - Apply one of eight strategies to select hard negatives.

4. DPO Fine-Tuning
    
    - Train models using HuggingFace-based DPO with selected negatives.

5. Evaluation

    - Assess performance on NDCG, HR, diversity, genre fairness, and over-recommendation.


## Setup
Moedl: HuggingFaceTB/SmolLM2-1.7B-Instruct

## Metrics

We report top-5 and top-10 metrics including:

    - Ranking metrics: NDCG@k, HR@k

    - Diversity: unique item count, DivRatio

    - Fairness: DGU, MGU

    - Over-recommendation: ORRatio

    - Validity: Predict_NotIn_Ratio

## Results Summery

### Comparative Analysis of Negative‑Sampling Methods on Goodreads

Negative Sampling Strategies

- clusterout_low: User-Preference-Clusters‑aware, sampling negatives outside interest clusters with low exposure.

- clusterin_high_clusterout_low: Hybrid—one high‑exposure negative inside the cluster + one low‑exposure negative outside.

SampleMethod | MGU@5 ↓ | DGU@5 ↓ | DivRatio@5 ↑ | ORRatio@5 ↓ | NDCG@5 ↑ | HR@5 ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| ClusterExposure-DPO(clusterout_low) | 0.0186| <mark>**0.0644**</mark> | <mark>**0.1366**</mark> | 0.0966 | <mark>**0.0211**</mark> | <mark>**0.0330**</mark> |
| ClusterExposure-DPO(clusterin_high_clusterout_low) | <mark>**0.0175**</mark> | 0.0650 | 0.1292 | 0.0946 | 0.0200 | 0.030 |
| SPRec(Baseline) | 0.0228 | 0.0789 | 0.1266 | <mark>**0.0870**</mark> | 0.0140 | 0.0220 |


- clusterout_low leads on diversity (DivRatio@5 0.1366) and ranking (NDCG@5 0.0211, HR@5 0.0330).
- clusterin_high_clusterout_low achieves the lowest MGU@5 (0.0175) and reduces ORRatio@5 compared to clusterout_low (0.0946 vs. 0.0966).
- SPRec still holds the best ORRatio@5 (0.0870) but lags in diversity and ranking.




| SampleMethod | MGU@10 ↓ | DGU@10 ↓ | DivRatio@10 ↑ | ORRatio@10 ↓ | NDCG@10 ↑ | HR@10 ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| ClusterExposure-DPO(clusterout_low) | 0.0129 | 0.0436 | <mark>**0.1026**</mark> | 0.0648 | <mark>**0.0230**</mark> | <mark>**0.039**</mark> |
| ClusterExposure-DPO(clusterin_high_clusterout_low) | <mark>**0.0113**</mark> | <mark>**0.0404**</mark> | 0.0999 | <mark>**0.0623**</mark> | 0.0216 | 0.035 |
| SPRec(Baseline) | 0.0141 | 0.0450 | 0.0968 | <mark>**0.0623**</mark> | 0.0158 | 0.028 |


- clusterout_low again yields the highest diversity (0.1026) and best NDCG/HR.

- clusterin_high_clusterout_low further lowers MGU@10 and DGU@10 (0.0113, 0.0404), matching the best ORRatio@10 (0.0623).

- SPRec retains best ORRatio but has the weakest diversity/ranking.

### Key Takeaways

- Reducing Bias: clusterin_high_clusterout_low is best at mitigating genre/popularity bias (MGU, DGU, ORRatio), especially at Top‑10.

- Maximizing Diversity & Accuracy: clusterout_low consistently leads in diversity (DivRatio) and retrieval quality (NDCG, HR).

- Baseline SPRec: Excels only on ORRatio but underperforms on diversity and ranking metrics.


## Experiments

### Negative Sampling Strategies

| Strategy Name | Description |
| --- | --- |
| balanced_popularity | Randomly select from high/low exposure halves |
| clusterin_high | In-cluster candidate with highest exposure |
| clusterin_low | In-cluster candidate with lowest exposure |
| low_exposure | Globally lowest exposure candidate |
| high_exposure | Globally highest exposure candidate |
| clusterout_low | Out-of-cluster candidate with lowest exposure |
| clusterout_high | Out-of-cluster candidate with highest exposure |
| clusterin_high_clusterout_low | One in-cluster high exposure + one out-of-cluster low exposure candidate |

### Top-5 Metrics

| SampleMethod | MGU@5 ↓ | DGU@5 ↓ | DivRatio@5 ↑ | ORRatio@5 ↓ | NDCG@5 ↑ | HR@5 ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| SPRec(Baseline) | 0.0228 | 0.0789 | 0.1266 | <mark>**0.0870**</mark> | 0.0140 | 0.022 |
| clusterout_high | 0.0185 | 0.0652 | 0.1300 | 0.1154 | 0.0151 | 0.023 |
| low_exposure | 0.0201 | 0.0743 | 0.1284 | 0.0986 | 0.0166 | 0.025 |
| high_exposure | 0.0187 | 0.0655 | 0.1366 | 0.1200 | 0.0150 | 0.023 |
| clusterin_high | 0.0187 | 0.0656 | <mark>**0.1368**</mark> | 0.1202 | 0.0150 | 0.023 |
| clusterout_low | 0.0186 | <mark>**0.0644**</mark> | 0.1366 | 0.0966 | <mark>**0.0211**</mark> | <mark>**0.033**</mark> |
| clusterin_low | 0.0196 | 0.0730 | 0.1292 | 0.0978 | 0.0170 | 0.027 |
| clusterin_high_clusterout_low | <mark>**0.0175**</mark> | 0.0650 | 0.1292 | 0.0946 | 0.0200 | 0.030 |
| balanced_popularity | 0.0183 | 0.0682 | 0.1296 | 0.0950 | 0.0194 | 0.029 |
| clusterin_low_clusterout_low | 0.0186 | 0.0708 | 0.1256 | 0.0896 | 0.0177 | 0.027 |
| LightGCN(all) | 0.0214 | 0.0749 | 0.1232 | 0.1146 | 0.0117 | 0.0190 |

### Top-10 Metrics

| SampleMethod | MGU@10 ↓ | DGU@10 ↓ | DivRatio@10 ↑ | ORRatio@10 ↓ | NDCG@10 ↑ | HR@10 ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| SPRec(Baseline) | 0.0228 | 0.0789 | 0.1266 | <mark>**0.0870**</mark> | 0.0140 | 0.022 | | 0.0141 | 0.0450 | 0.0968 | <mark>**0.0623**</mark> | 0.0158 | 0.028 |
| clusterout_high | 0.0125 | 0.0418 | 0.0979 | 0.0807 | 0.0167 | 0.028 |
| low_exposure | 0.0127 | 0.0452 | 0.0983 | 0.0656 | 0.0182 | 0.030 |
| high_exposure | 0.0129 | 0.0435 | 0.1027 | 0.0849 | 0.0165 | 0.028 |
| clusterin_high | 0.0129 | 0.0435 | <mark>**0.1032**</mark> | 0.0850 | 0.0165 | 0.028 |
| clusterout_low | 0.0129 | 0.0436 | 0.1026 | 0.0648 | <mark>**0.0230**</mark> | <mark>**0.039**</mark> |
| clusterin_low | 0.0125 | 0.0443 | 0.0978 | 0.0655 | 0.0186 | 0.032 |
| clusterin_high_clusterout_low | <mark>**0.0113**</mark> | <mark>**0.0404**</mark> | 0.0999 | <mark>**0.0623**</mark> | 0.0216 | 0.035 |
| balanced_popularity | 0.0119 | 0.0417 | 0.0983 | 0.0649 | 0.0214 | 0.035 |
| clusterin_low_clusterout_low | 0.0125 | 0.0457 | 0.0951 | 0.0626 | 0.0196 | 0.033 |
| LightGCN(all) | 0.0131 | 0.0413 | 0.0957 | 0.0787 | 0.01355 | 0.0250 |

### Predict Not-In-Ratio

| SampleMethod | Predict_NotIn_Ratio ↓ |
| --- | --- |
| SPRec(Baseline) |  <mark>**0.0623**</mark> | 
| clusterout_high | 0.8150 |
| low_exposure | 0.7490 |
| high_exposure | 0.8250 |
| clusterin_high | 0.8250 |
| clusterout_low | 0.7790 |
| clusterin_low | 0.7480 |
| clusterin_high_clusterout_low | 0.8000 |
| balanced_popularity | 0.7870 |
| clusterin_low_clusterout_low | 0.7700 |


## Quick Start
### ClusterExposure
1. scripts/run_sft.py
2. scripts/run_beam_cd_candidates.py
3. scripts/run_annotate_candidates.py
4. scripts/run_cluster_exposure_neg_sampling.py
5. scripts/run_all_dpo.py
6. scripts/run_generate_predictions.py
7. scripts/run_evaluate.py

### SPRec_Reproduction
1. scripts/run_sft.py
2. scripts/run_generate_selfplay.py
3. scripts/run_dpo.py
4. scripts/run_generate_predictions.py
5. scripts/run_evaluate.py

## Directory Structure

```
ClusterExposure-DPO/
├── configs/                 # Config files (YAML)
├── scripts/                 # Entry points for each stage
├── src/
│   ├── dataset/             # Beam search, annotation, and sampling
│   ├── evaluation/          # Metric evaluation logic
│   └── models/              # DPO/SFT/SDPO training logic
├── experiments/
│   ├── data/                # Generated training/valid data
│   ├── model/               # Trained model checkpoints
│   ├── predictions/         # Inference outputs
│   └── metrics/             # Evaluation results
```