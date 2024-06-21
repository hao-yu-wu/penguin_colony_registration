import os
import torch

import numpy as np
from PIL import Image
import cv2
from torchvision.transforms.functional import to_pil_image

from geometry import render_top
from loguru import logger

from render import fragments2depth, prepare_renderer, sam_post, blend_vis, _parse_xml_geo
from metrics import meanIOU, shape_distance, areaRatio, pixelAcc

if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=2 python metrics2.py 

    colony_name = 'Devil_Island'
    # colony_name = 'Brown_Bluff'

    device = "cuda"
    root_dir = './ATA'
    ref_name = 'ref'
    eval_mode = 'confidence_interval'

    if eval_mode in ['confidence_interval', 'prompts15']:
        json_str_list = [f'{root_dir}/{colony_name}/runs/xxxxx_{run_i}.json' for run_i in range(30)]
    elif eval_mode == 'prompts3':
        json_str_list = [f'{root_dir}/{colony_name}/prompts/3/xxxxx.json']
    elif eval_mode == 'prompts9':
        json_str_list = [f'{root_dir}/{colony_name}/prompts/9/xxxxx.json']
    elif eval_mode == 'prompts12':
        json_str_list = [f'{root_dir}/{colony_name}/prompts/12/xxxxx.json']

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

    mIoUs, PAR_Distances, Area_Errors, Accuracys = [], [], [], []
    mIoUs_bird, PAR_Distances_bird, Area_Errors_bird, Accuracys_bird = [], [], [], []

    for json_str in json_str_list:
        """------------------------------------------------------ render------------------------------------------------------------"""
        xml_path_ref = f'{root_dir}/{colony_name}/{ref_name}.xml'
        mesh_path = f'{root_dir}/{colony_name}/data.ply'
        json_file_ref = f'{root_dir}/{colony_name}/{ref_name}.json'
        if not os.path.exists(json_file_ref): json_file_ref=None
        
        # top view, reference image: fixed resolution.
        if xml_path_ref is not None:
            the_renderer, the_data, the_cameras, start_R, start_T = prepare_renderer(device, mesh_path, xml_path_ref, ground_file=None)
            images, fragments = the_renderer(the_data, cameras=the_cameras, R=start_R, T=start_T)
            depth = fragments2depth(fragments) # [1, H, W]
            depth_ref = depth[0, ...] # [H, W]
            rgba_ref = images.permute(0, 3, 1, 2).clone() # (1, C, H, W)
        if json_file_ref is not None:
            rgba_ref_mask = sam_post(json_file_ref) 
            rgba_ref_mask = torch.from_numpy(rgba_ref_mask[None,None,:,:]).to(device)

        front = torch.zeros_like(rgba_ref[:,:3])
        front_mask = torch.zeros_like(rgba_ref[:,3:])
        for img_name in img_names:
            # find the image
            for _ext in ['png', 'jpg', 'JPG', 'jpeg', 'PNG']:
                ground_file = f'{root_dir}/{colony_name}/{img_name}.{_ext}'
                if os.path.exists(ground_file):
                    break
            assert os.path.exists(ground_file), f'{ground_file} does not exist'
            # find mask & camera pose file
            json_file = json_str.replace('xxxxx', img_name) # segmentation mask
            xml_path = f'{root_dir}/{colony_name}/{img_name}.xml' # camera pose

            # rendered src view, resize to have same height and larger width with ground image in src view.
            the_renderer, the_data, the_cameras, start_R, start_T = prepare_renderer(device, mesh_path, xml_path, ground_file)

            # render
            images, fragments = the_renderer(the_data, cameras=the_cameras, R=start_R, T=start_T)
            # h, w
            render_h, render_w = images[0, ..., 3].shape
            rgba_rendered = images.permute(0, 3, 1, 2).clone()
            alpha_mask = images[0, ..., 3].cpu().numpy() # [H, W]
            depth = fragments2depth(fragments) # [1, H, W]
            depth_np = depth[0, ...].cpu().numpy() # [H, W]
            images_np = (images[0, ..., :3]).cpu().numpy().clip(0.,1.) # [H, W, 3]

            dmi, dma = depth[depth>0].min(), depth[depth>0].max()
            depth_gray = ((depth-dmi) / (dma-dmi + 1e-8)).clip(0.,1.)
            # sam.mask
            sam_mask = sam_post(json_file)
            sam_mask = cv2.resize(sam_mask, (render_w, render_h))
            sam_mask = sam_mask > 0
            with Image.open(ground_file) as f:
                ground_Img = np.asarray(f, dtype=np.float32) / 255.
                # remove alpha channel (i.e., set to 1)
                ground_Img = ground_Img[:,:,:3]
                # h, w, _ = ground_Img.shape
                ground_Img = cv2.resize(ground_Img, (render_w, render_h))
                ground_Img = torch.from_numpy(ground_Img[None])
            # mask sam
            sam_mask &= (alpha_mask > 0) & (depth_np > 0)
            sam_mask = torch.from_numpy(sam_mask[None,:,:,None].astype(np.float32))
            sam_image = ground_Img * sam_mask
            sam_image = torch.cat([sam_image, sam_mask], dim=-1)
            sam_image = sam_image.to(device)

            pose_src, intrinsics_src = _parse_xml_geo(xml_path, ground_file)
            pose_ref, intrinsics_ref = _parse_xml_geo(xml_path_ref, ground_file=None)
            rgba_src = sam_image.permute(0, 3, 1, 2).clone()
            depth_src = depth[0].clone()
            # blended - src view
            blended_src_view = blend_vis(rgba_src, rgba_rendered, rgba_src[:,3:], white_width=3)
            # blended - ref view
            sampled_rgba_src = render_top(rgba_src, pose_ref, pose_src, intrinsics_ref, intrinsics_src, depth_ref, depth_src=depth_src)
            if json_file_ref is not None:
                sampled_rgba_src *= rgba_ref_mask # w ref mask
            blended_ref_view = blend_vis(sampled_rgba_src, rgba_ref, sampled_rgba_src[:,3:], white_width=3)

            # RGB += (1-M) * (new_RGB * new_M)
            # M += (1-M) * new_M
            front += (1-front_mask) * sampled_rgba_src[:,:3]
            front_mask += (1-front_mask) * sampled_rgba_src[:,3:]

            logger.info(f'-------{json_file}: end------------')

        # blended - ref view (total)
        logger.info(f'-------blended - ref view (total)------------')
        final = blend_vis(front, rgba_ref, front_mask, white_width=3)
        pred_mask_pil = to_pil_image(front_mask[0,...].cpu())

        """------------------------------------------------------ metrics------------------------------------------------------------"""
        gt_dir = f'./ATA/{colony_name}-GT'

        pred_path = [json_str.replace('xxxxx', img_name) for img_name in img_names]
        gt_path = [os.path.join(gt_dir, img_name+'.json') for img_name in img_names]

        pred_masks = [sam_post(name) for name in pred_path]
        gt_masks = [sam_post(name) for name in gt_path]

        mIoU_i = meanIOU(gt_masks, pred_masks)
        PAR_Distance_i = shape_distance(gt_masks, pred_masks)
        Area_Error_i = areaRatio(gt_masks, pred_masks)
        Accuracy_i = pixelAcc(gt_masks, pred_masks)

        mIoUs.append(mIoU_i)
        PAR_Distances.append(PAR_Distance_i)
        Area_Errors.append(Area_Error_i)
        Accuracys.append(Accuracy_i)

        if colony_name == 'Devil_Island':
            gt_mask_path = f"./ATA/{colony_name}/ref_mask.png"
            gt_mask = np.array(Image.open(gt_mask_path))
            gt_mask = gt_mask[...,0] > 0

            pred_mask = np.array(pred_mask_pil)
            pred_mask = pred_mask > 0
            assert gt_mask.shape == pred_mask.shape # run render.py with ref & Devil_Island

            gt_masks = [gt_mask.astype(np.int64)]
            pred_masks = [pred_mask.astype(np.int64)]

            mIoU_bird_i = meanIOU(gt_masks, pred_masks)
            PAR_Distance_bird_i = shape_distance(gt_masks, pred_masks)
            Area_Error_bird_i = areaRatio(gt_masks, pred_masks)
            Accuracy_bird_i = pixelAcc(gt_masks, pred_masks)

            mIoUs_bird.append(mIoU_bird_i)
            PAR_Distances_bird.append(PAR_Distance_bird_i)
            Area_Errors_bird.append(Area_Error_bird_i)
            Accuracys_bird.append(Accuracy_bird_i)
    
    print("------------2D segmentations------------")
    for the_metric in [mIoUs, PAR_Distances, Area_Errors, Accuracys]:
        np_metric = np.array(the_metric)
        print(np_metric.mean(), np_metric.std()/(len(np_metric)**0.5)*1.96)

    if colony_name == 'Devil_Island':
        print("------------bird-eye-view metrics for devi (only devi has GT to eval)---------------")
        for the_metric in [mIoUs_bird, PAR_Distances_bird, Area_Errors_bird, Accuracys_bird]:
            np_metric = np.array(the_metric)
            print(np_metric.mean(), np_metric.std()/(len(np_metric)**0.5)*1.96)