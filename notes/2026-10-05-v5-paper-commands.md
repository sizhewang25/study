# Analysis v5 paper command manuals

## Step 0: Prepare a group config in config/groups/

e.g., group id: pro-paper 

## Step 1: Get bounds of pooled datasets

Purpose: distance and RTT normlization

python -m scripts.analysis.v5.cli report-bounds --group pro-paper  

outputs/analysis/v5/_cross/bounds/pro-paper.seen/bounds.json

```
  "computed": {
    "dist_norm_km_max": 4387.257,
    "rtt_norm_ms_max": 92.395
  },
```

Then copy back to pro-paper configs:

```
analysis:
  common:
    dist_norm_km:
      min: 0.0
      max: 4387.257
    rtt_norm_ms:
      min: 0.0
      max: 92.395
```

## Step 2: Prerequisites per dataset

```
for r in pro-as0{1,2,3}-{mesh,loso}; do
  python -m scripts.analysis.v5.cli build-answer-space --run-id $r   # sites, seeds, pixels → answer-space/healpix-128/{meta.json,sites.csv,…}
  python -m scripts.analysis.v5.cli classify --run-id $r             # per-TG labels → classify/healpix-128/<method>_tgs.parquet (holds each TG's fold)
done
```

We get per-target classification.


## Step 3: Get dataset stats

python -m scripts.analysis.v5.cli report-dataset --group pro-paper

outputs/analysis/v5/_cross/dataset/pro-paper/dataset.csv
outputs/analysis/v5/_cross/dataset/pro-paper/dataset.manifest.json

