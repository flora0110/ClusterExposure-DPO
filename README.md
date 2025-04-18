# ClusterExposure-DPO

## results

### Top-5 Metrics
| SampleMethod | NDCG@5 | HR@5 | DivRatio@5 | DGU@5 | MGU@5 | ORRatio@5 |
| --- | --- | --- | --- | --- | --- | --- |
| SelfPaly | 0.0139822435636894 | 0.022 | 0.1266 | 0.0788942384583304 | 0.0227651259748188 | 0.087 |
| clusterout_high | 0.0150953907564549 | 0.023 | 0.13 | 0.0652478880052406 | 0.0185014477402672 | 0.1154 |
| low_exposure | 0.0166131733172609 | 0.025 | 0.1284 | 0.0742725943904345 | 0.0201182226040503 | 0.0986 |
| high_exposure | 0.0149567438726017 | 0.023 | 0.1366 | 0.0655017096017437 | 0.0186978353073167 | 0.12 |
| clusterin_high | 0.0149567438726017 | 0.023 | 0.1368 | 0.0655792603469052 | 0.0186727878616745 | 0.1202 |
| clusterout_low | 0.0211054561869791 | 0.033 | 0.1366 | 0.064402053598854 | 0.018647302065405 | 0.0966 |
| clusterin_low | 0.0170438498753343 | 0.027 | 0.1292 | 0.0729685741267756 | 0.0196360336386704 | 0.0978 |

### Top-10 Metrics

| SampleMethod | NDCG@10 | HR@10 | DivRatio@10 | DGU@10 | MGU@10 | ORRatio@10 |
| --- | --- | --- | --- | --- | --- | --- |
| SelfPaly | 0.0158400085587484 | 0.028 | 0.0968 | 0.0450313294761574 | 0.0141333675665805 | 0.0623 |
| clusterout_high | 0.0166760560945421 | 0.028 | 0.0979 | 0.0418171370629533 | 0.0124714201432745 | 0.0807 |
| low_exposure | 0.0182082735364698 | 0.03 | 0.0983 | 0.0451840861975525 | 0.0126790831371458 | 0.0656 |
| high_exposure | 0.0165374092106889 | 0.028 | 0.1027 | 0.0434535172913557 | 0.0129316413948564 | 0.0849 |
| clusterin_high | 0.0165374092106889 | 0.028 | 0.1032 | 0.0434921247375571 | 0.0129340393728814 | 0.085 |
| clusterout_low | 0.022989621232506 | 0.039 | 0.1026 | 0.0435878040607508 | 0.0129244474607813 | 0.0648 |
| clusterin_low | 0.0186389500945432 | 0.032 | 0.0978 | 0.0443448955081479 | 0.0124505041698884 | 0.0655 |

### Predict Not-In-Ratio

| SampleMethod | Predict_NotIn_Ratio |
| --- | --- |
| SelfPaly | 0.701 |
| clusterout_high | 0.815 |
| low_exposure | 0.749 |
| high_exposure | 0.825 |
| clusterin_high | 0.825 |
| clusterout_low | 0.779 |
| clusterin_low | 0.748 |

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
