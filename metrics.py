import numpy as np
import json
import cv2
import os
from PIL import Image
from skimage import measure
import random

def sam_post(json_file):
    """
    return (H, W) 0..1 np.float32
    """
    # Load JSON file
    with open(json_file) as f:
        data = json.load(f)

    # Initialize an empty image with the required dimensions
    # The dimensions of the output mask should match the dimensions of the image
    # the JSON file corresponds to. You may need to adjust these values.
    height, width = data["imageHeight"], data["imageWidth"]  # Adjust these dimensions to match your specific case
    img = np.zeros([height, width], dtype=np.uint8)

    # Iterate through shapes and fill polygons
    for shape in data['shapes']:
        polygon = np.array(shape['points'], np.int32)
        cv2.fillPoly(img, [polygon], color=(255))

    # print(img.shape, type(img), img.max()) # (450, 800) <class 'numpy.ndarray'> 255
    return (img.astype(np.float32) / 255.).clip(0., 1.).astype(np.int64)

def meanIOU(target, predicted):
    N = len(target)
    
    iousum = 0
    for i in range(N):
        target_arr = target[i]
        predicted_arr = predicted[i]
        intersection = np.logical_and(target_arr, predicted_arr).sum()
        union = np.logical_or(target_arr, predicted_arr).sum()
        if union == 0:
            iou_score = 0
        else :
            iou_score = intersection / union
        iousum += iou_score
        
    miou = iousum/N
    return miou

def pixelAcc(target, predicted):    
    N = len(target)
    
    accsum=0
    for i in range(N):
        target_arr = target[i]
        predicted_arr = predicted[i]
        
        same = (target_arr == predicted_arr).sum()
        a, b = target_arr.shape
        total = a*b
        accsum += same/total
    
    pixelAccuracy = accsum/N      
    return pixelAccuracy

def areaRatio(target, predicted):
    N = len(target)
    
    res=0
    for i in range(N):
        # print(target[i].shape, target[i].dtype, target[i].min(), target[i].max()) # (H, W) int64  0  1
        target_arr = target[i]
        predicted_arr = predicted[i]
        
        ratio = abs(predicted_arr.sum() - target_arr.sum()) / target_arr.sum()
        res += ratio
    res = res/N      
    return res

def shape_distance(gt_masks, pred_masks):
    res = 0
    N = len(pred_masks)

    contours_pred = []
    for pred in pred_masks:
        contour1 = measure.find_contours(pred, 0.5)
        contours_pred.append(contour1)
        
    contours_gt = []
    for gt in gt_masks:
        contour1 = measure.find_contours(gt, 0.5)
        contours_gt.append(contour1)

    for c in range(len(pred_masks)):
        perimeters = []
        areas = []
        for contour in contours_pred[c]:
            c1 = np.expand_dims(contour.astype(np.float32), 1)
            c1 = cv2.UMat(c1)
            area = cv2.contourArea(c1)
            areas.append(area)
            perimeter = cv2.arcLength(c1, True)
            perimeters.append(perimeter)
        perimeters2 = []
        areas2 = []
        for contour in contours_gt[c]:
            c1 = np.expand_dims(contour.astype(np.float32), 1)
            c1 = cv2.UMat(c1)
            area = cv2.contourArea(c1)
            areas2.append(area)
            perimeter = cv2.arcLength(c1, True)
            perimeters2.append(perimeter)
        sum_per = sum(perimeters)
        sum_areas = sum(areas)
        sum_per2 = sum(perimeters2)
        sum_areas2 = sum(areas2)
        per_area = float(sum_per)/float(sum_areas)
        per_area2 = float(sum_per2)/float(sum_areas2)
        # print("--------------------------------------------------")
        # print("Overall perimeter/area ratio for pred ", c, " = ", np.format_float_positional(per_area, precision=4, unique=False, fractional=False, trim='k'))
        # print("Overall perimeter/area ratio for gt ", c, " = ", np.format_float_positional(per_area2, precision=4, unique=False, fractional=False, trim='k'))
        # print("Overall perimeter/area ratio difference between pred and gt = ", np.format_float_positional(np.abs(per_area - per_area2), precision=4, unique=False, fractional=False, trim='k'))
        # print("--------------------------------------------------")  
        res += np.abs(per_area - per_area2)  

    res = res / N
    return res

if __name__ == "__main__":
    colony_name = 'Devil_Island'
    # colony_name = 'Brown_Bluff'

    print(f'-----------{colony_name}--------')

    if colony_name == 'Brown_Bluff':
        img_names = [
            '2Brown Bluff Edge of Adelie Colony 450 ancho',
            '2009_02_05_9534-antarctica_brown-bluff',
            '33091353036_c128403f54_o',
            '15850340520_04b36d6e8c_o',

            'BROW_073',
            'BROW_FEB96_N113_Slide26',
            'BROW_FEB96_N113_Slide27',
            'BROW_FEB96_N113_Slide28',
            'BROW_FEB96_N113_Slide36',
        ]
    else:
        img_names = [
            '4- Group C from Photopoint 4',
            '10- Group A from Photopoint 4',
            'devi_ground_photo_1',
            'Devil_Island0607-001',
            'Devil_Island0607-006',
            'DEVI_2_076',
            
            '00',
            'DEVI_1_081',
        ]

    pred_dir = f'./ATA/{colony_name}'
    gt_dir = f'./ATA/{colony_name} - GT'

    pred_path = [os.path.join(pred_dir, name+'.json') for name in img_names]
    gt_path = [os.path.join(gt_dir, name+'.json') for name in img_names]

    pred_masks = [sam_post(name) for name in pred_path]
    gt_masks = [sam_post(name) for name in gt_path]

    print("------------2D segmentations------------")
    print('mIoU', meanIOU(gt_masks, pred_masks))
    print('PAR Distance', shape_distance(gt_masks, pred_masks))
    print('Area Error', areaRatio(gt_masks, pred_masks))
    print('Accuracy', pixelAcc(gt_masks, pred_masks))

    # pred_mask = sam_post("ATA/Brown_Bluff/15850340520_04b36d6e8c_o.json")
    # print(pred_mask.shape, pred_mask.dtype, pred_mask.max(), pred_mask.sum())

    """-------------------------bird-eye-view metrics---------------"""
    if colony_name == 'Devil_Island':
        gt_mask_path = f"./ATA/{colony_name}/ref_mask.png"
        pred_mask_path = f"./ATA/{colony_name}/results/pred_mask.png"
        gt_mask = np.array(Image.open(gt_mask_path))
        gt_mask = gt_mask[...,0] > 0
        # print(gt_mask.shape, gt_mask.dtype, gt_mask.max(), gt_mask.min()) # (1308, 2444) int64 1 0

        pred_mask = np.array(Image.open(pred_mask_path))
        pred_mask = pred_mask > 0
        # Image.fromarray(gt_mask & pred_mask).save('./debug.png')
        assert gt_mask.shape == pred_mask.shape # run render.py with ref_whole & Devil_Island

        gt_masks = [gt_mask.astype(np.int64)]
        pred_masks = [pred_mask.astype(np.int64)]

        print("------------bird-eye-view metrics for devi (only devi has GT to eval)---------------")
        print('mIoU', meanIOU(gt_masks, pred_masks))
        print('PAR Distance', shape_distance(gt_masks, pred_masks))
        print('Area Error', areaRatio(gt_masks, pred_masks))
        print('Accuracy', pixelAcc(gt_masks, pred_masks))