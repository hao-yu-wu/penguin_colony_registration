# Penguin colony georegistration using camera pose estimation and phototourism 

## Env Steup
Tested with CUDA 11.3, Ubuntu 20.04
```bash
conda env create -f environment.yml
conda activate eg3d
```

## Data
The data are in ```./ATA``` folder, with its structure below.

The licenses of the data are in ```wu_et_al_appendix_S1.docx```.
```
./ATA
├── Devil_Island
│   |── data.ply (mesh derived from DEM and satellite image)
│   |── image1.json (segmentation - polygons)
│   |── image2.json 
│   |── ... 
│   |── image1.png (image)
│   |── image2.png
│   |── ... 
│   |── image1.xml (camera pose)
│   |── image2.xml
│   |── ... 
│   |── ref.json (bird-eye view segmentation - polygons)
│   |── ref_mask.png (bird-eye view segmentation - mask)
│   |── ref.xml (bird-eye view camera pose)

└──  Devil_Island-GT (manual segmentation - polygons)
│   |── image1.json
│   |── image2.json 
│   └── ...

├── Brown_Bluff
│   └── ...

└── Brown_Bluff-GT
    └── ...
```

## Method

### Segmentation
We use [AnyLabeling](https://github.com/vietanhdev/anylabeling) ("Segment Anything (ViT-H Quant)") to create segmentation masks. 

If you want to recreate some masks by yourself, you can download AnyLabeling [here (windows exe file)](https://github.com/vietanhdev/anylabeling/releases/download/v0.3.3/AnyLabeling.exe).

### Registration
```bash
# set colony_name = Devil_Island/Brown_Bluff
CUDA_VISIBLE_DEVICES=0 python render.py 
```
What does ```render.py``` do?
1. use the segmentation mask (json) to segment, save result -> ```./ATA/{colony_name}/results```
2. render top bird-eye view with everything together -> ```./ATA/{colony_name}/results/pred_mask.png```

### Compute Metrics
```bash
# set colony_name = Devil_Island/Brown_Bluff
# set test_gt=True for fully manual method
python metrics.py

# metrics with confidence interval or sensitivity analysis
# set eval_mode = confidence_interval/prompts3/promps9/prompts12/prompts15
CUDA_VISIBLE_DEVICES=0 python metrics2.py
```

## Contact
- haoyuwu@cs.stonybrook.edu
- heather.lynch@stonybrook.edu