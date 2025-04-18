# ClusterExposure-DPO

## results

### Top-5 Metrics

| Model | NDCG@5 | HR@5 | DivRatio@5 | DGU@5 | MGU@5 | ORRatio@5 |
| --- | --- | --- | --- | --- | --- | --- |
| SPRec                 | 0.0140 | 0.022 | 0.1266 | 0.0789 | 0.0228 | 0.0870 |
| ClusterExposure_model | 0.0151 | 0.023 | 0.1300 | 0.0652 | 0.0185 | 0.1154 |
| ClusterExposure_model | 0.0166 | 0.025 | 0.1284 | 0.0743 | 0.0201 | 0.0986 |
| ClusterExposure_model | 0.0150 | 0.023 | 0.1366 | 0.0655 | 0.0187 | 0.1200 |
| ClusterExposure_model | 0.0150 | 0.023 | 0.1368 | 0.0656 | 0.0187 | 0.1202 |
| ClusterExposure_model | 0.0211 | 0.033 | 0.1366 | 0.0644 | 0.0186 | 0.0966 |
| ClusterExposure_model | 0.0170 | 0.027 | 0.1292 | 0.0730 | 0.0196 | 0.0978 |

### Top-10 Metrics

| Model | NDCG@10 | HR@10 | DivRatio@10 | DGU@10 | MGU@10 | ORRatio@10 |
| --- | --- | --- | --- | --- | --- | --- |
| SPRec                 | 0.0158 | 0.028 | 0.0968 | 0.0450 | 0.0141 | 0.0623 |
| ClusterExposure_model | 0.0167 | 0.028 | 0.0979 | 0.0418 | 0.0125 | 0.0807 |
| ClusterExposure_model | 0.0182 | 0.030 | 0.0983 | 0.0452 | 0.0127 | 0.0656 |
| ClusterExposure_model | 0.0165 | 0.028 | 0.1027 | 0.0435 | 0.0129 | 0.0849 |
| ClusterExposure_model | 0.0165 | 0.028 | 0.1032 | 0.0435 | 0.0129 | 0.0850 |
| ClusterExposure_model | 0.0230 | 0.039 | 0.1026 | 0.0436 | 0.0129 | 0.0648 |
| ClusterExposure_model | 0.0186 | 0.032 | 0.0978 | 0.0443 | 0.0125 | 0.0655 |

### Predict Not-In-Ratio

| Model | Predict_NotIn_Ratio |
| --- | --- |
| SPRec | 0.701 |
| ClusterExposure_model | 0.815 |
| ClusterExposure_model | 0.749 |
| ClusterExposure_model | 0.825 |
| ClusterExposure_model | 0.825 |
| ClusterExposure_model | 0.779 |
| ClusterExposure_model | 0.748 |

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
