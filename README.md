# Penguin colony georegistration using camera pose estimation and phototourism 

## Env Steup
```bash
conda env create -f environment.yml
conda activate eg3d
```

## Download Data
Please email haoyuwu@cs.stonybrook.edu for downloading data.

## Run Method
1. use the segmentation mask (json) to segment, save result -> ./ATA/Devil_Island/results
2. render top bird-eye view with everything together
```bash
# set colony_name: Devil_Island/Brown_Bluff
CUDA_VISIBLE_DEVICES=0 python render.py 
```

## Compute Metrics
```bash
# set colony_name: Devil_Island/Brown_Bluff
python metrics.py
```