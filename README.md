# ClusterExposure-DPO

## results

### Top-5 Metrics

| SampleMethod | MGU@5 ↓ | DGU@5 ↓ | DivRatio@5 ↑ | ORRatio@5 ↓ | NDCG@5 ↑ | HR@5 ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| SelfPaly | 0.0228 | 0.0789 | 0.1266 | <mark>**0.0870**</mark> | 0.0140 | 0.022 |
| clusterout_high | 0.0185 | 0.0652 | 0.1300 | 0.1154 | 0.0151 | 0.023 |
| low_exposure | 0.0201 | 0.0743 | 0.1284 | 0.0986 | 0.0166 | 0.025 |
| high_exposure | 0.0187 | 0.0655 | 0.1366 | 0.1200 | 0.0150 | 0.023 |
| clusterin_high | 0.0187 | 0.0656 | <mark>**0.1368**</mark> | 0.1202 | 0.0150 | 0.023 |
| clusterout_low | 0.0186 | <mark>**0.0644**</mark> | 0.1366 | 0.0966 | <mark>**0.0211**</mark> | <mark>**0.033**</mark> |
| clusterin_low | 0.0196 | 0.0730 | 0.1292 | 0.0978 | 0.0170 | 0.027 |
| clusterin_high_clusterout_low | <mark>**0.0175**</mark> | 0.0650 | 0.1292 | 0.0946 | 0.0200 | 0.030 |
| balanced_popularity | 0.0183 | 0.0682 | 0.1296 | 0.0950 | 0.0194 | 0.029 |
| clusterin_low_clusterout_low | 0.0186 | 0.0708 | 0.1256 | 0.0896 | 0.0177 | 0.027 |


### Top-10 Metrics

| SampleMethod | MGU@10 ↓ | DGU@10 ↓ | DivRatio@10 ↑ | ORRatio@10 ↓ | NDCG@10 ↑ | HR@10 ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| SelfPaly | 0.0141 | 0.0450 | 0.0968 | <mark>**0.0623**</mark> | 0.0158 | 0.028 |
| clusterout_high | 0.0125 | 0.0418 | 0.0979 | 0.0807 | 0.0167 | 0.028 |
| low_exposure | 0.0127 | 0.0452 | 0.0983 | 0.0656 | 0.0182 | 0.030 |
| high_exposure | 0.0129 | 0.0435 | 0.1027 | 0.0849 | 0.0165 | 0.028 |
| clusterin_high | 0.0129 | 0.0435 | <mark>**0.1032**</mark> | 0.0850 | 0.0165 | 0.028 |
| clusterout_low | 0.0129 | 0.0436 | 0.1026 | 0.0648 | <mark>**0.0230**</mark> | <mark>**0.039**</mark> |
| clusterin_low | 0.0125 | 0.0443 | 0.0978 | 0.0655 | 0.0186 | 0.032 |
| clusterin_high_clusterout_low | <mark>**0.0113**</mark> | <mark>**0.0404**</mark> | 0.0999 | 0.0623 | 0.0216 | 0.035 |
| balanced_popularity | 0.0119 | 0.0417 | 0.0983 | 0.0649 | 0.0214 | 0.035 |
| clusterin_low_clusterout_low | 0.0125 | 0.0457 | 0.0951 | 0.0626 | 0.0196 | 0.033 |


### Predict Not-In-Ratio

| SampleMethod | Predict_NotIn_Ratio ↓ |
| --- | --- |
| SelfPaly | <mark>**0.7010**</mark> |
| clusterout_high | 0.8150 |
| low_exposure | 0.7490 |
| high_exposure | 0.8250 |
| clusterin_high | 0.8250 |
| clusterout_low | 0.7790 |
| clusterin_low | 0.7480 |
| clusterin_high_clusterout_low | 0.8000 |
| balanced_popularity | 0.7870 |
| clusterin_low_clusterout_low | 0.7700 |


## ClusterExposure
1. scripts/run_sft.py
2. scripts/run_beam_cd_candidates.py
3. scripts/run_annotate_candidates.py
4. scripts/run_cluster_exposure_neg_sampling.py
5. scripts/run_all_dpo.py
6. scripts/run_generate_predictions.py
7. scripts/run_evaluate.py

## SPRec_Reproduction
1. scripts/run_sft.py
2. scripts/run_generate_selfplay.py
3. scripts/run_dpo.py
4. scripts/run_generate_predictions.py
5. scripts/run_evaluate.py
