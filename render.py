import copy
import os
import xml.etree.ElementTree as ET
import torch
from matplotlib import pyplot as plt
import numpy as np
from torch import nn 
import open3d as o3d
from PIL import Image
import json
import cv2
from torchvision.transforms.functional import to_pil_image

from geometry import render_top
from loguru import logger

from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings, 
    # MeshRenderer, 
    MeshRasterizer,  
    HardPhongShader,
    PerspectiveCameras,
    TexturesVertex,
    PointLights,
)
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams


class SimpleShader(nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image

def _parse_xml_geo(file_name, ground_file):
    if ground_file is not None:
        with Image.open(ground_file) as f:
            ground_Img = np.asarray(f, dtype=np.float32) / 255.
            h, w, _ = ground_Img.shape
            wh_ratio = w*1. / (h*1.)

    inFile = ET.parse(file_name)
    root = inFile.getroot()

    for camera in root:
        if camera.tag != 'VCGCamera': continue

        # Translation vector
        translationVector = camera.attrib['TranslationVector']
        # Distortion
        distortion = camera.attrib['LensDistortion']
        # ViewPort
        size = camera.attrib['ViewportPx']
        # PixelSize
        pixsize = camera.attrib['PixelSizeMm']
        # CenterPx
        centerpix = camera.attrib['CenterPx']
        # Focal lenght
        focal = camera.attrib['FocalMm']
        # Rotation matrix
        rotation = camera.attrib['RotationMatrix']

        # To obtain the correct value for the focal length
        # we also need the pixel sizes
        focal = float(focal)
        pixsize = pixsize.split()
        focalX = focal / float(pixsize[0])
        focalY = focal / float(pixsize[1])

        # Image Dimensions
        imsize = size.split()
        imWidth = int(imsize[0])
        imHeight = int(imsize[1])
        # Pixel Center
        pixcenter = centerpix.split()
        pixCenterX = int(pixcenter[0])
        pixCenterY = int(pixcenter[1])
        # clip width of im
        if ground_file is not None:
            assert imWidth >= int(wh_ratio * imHeight), 'meshlab always height first, clip width of the image or pad with 0 (empty)'
            imWidth = int(wh_ratio * imHeight)
        pixCenterX, pixCenterY = imWidth * 0.5, imHeight * 0.5

        # Rotation
        rotation = rotation.split()
        extrinsic = []
        for i in range(0, len(rotation)):
            extrinsic.append(float(rotation[i]))

        # Position
        position = []
        translationVector = translationVector.split()
        for i in range(0,3):
            position.append(float(translationVector[i]))

        R = np.eye(3, dtype=np.float32)
        R[0, 0], R[0, 1], R[0, 2] = extrinsic[0], extrinsic[1], extrinsic[2]
        R[1, 0], R[1, 1], R[1, 2] = extrinsic[4], extrinsic[5], extrinsic[6]
        R[2, 0], R[2, 1], R[2, 2] = extrinsic[8], extrinsic[9], extrinsic[10]

        T = np.zeros((3), dtype=np.float32)
        T[0], T[1], T[2] = position[0], position[1], position[2]

        # see render_top in geometry.py
        R_pytorch3d = R.T
        T_pytorch3d = T.reshape(1, 3) @ R.T
        R0 = np.eye(3, dtype=np.float32)
        R0[1,1], R0[2,2] = -1., -1. # meshlab format to conventional [R|T]!
        R_pytorch3d = R_pytorch3d @ R0
        T_pytorch3d = T_pytorch3d @ R0

        P = np.eye(4, dtype=np.float32)
        P[:3, :3] = copy.deepcopy(R_pytorch3d.T)
        P[:3, 3:] = copy.deepcopy(T_pytorch3d.T)

        K = np.array([focalX, focalY, pixCenterX, pixCenterY])

    return P, K

def parse_xml(file_name, verts, ground_file, in_ndc=False):
    """
    meshlab
        X_c.T (3,1) = R @ (X_w.T + T) = R @ X_w.T + R @ T
        X_c   (1,3) = X_w @ R.T + T.reshape(1,3) @ R.T (1,3)
    pytorch3d
        X_c.T (4,1) = P.T @ X_w.T (4,1)
        X_c   (1,4) = X_w @ P     (1,4)
    """
    if ground_file is not None:
        with Image.open(ground_file) as f:
            ground_Img = np.asarray(f, dtype=np.float32) / 255.
            h, w, _ = ground_Img.shape
            wh_ratio = w*1. / (h*1.)

    inFile = ET.parse(file_name)
    root = inFile.getroot()

    for camera in root:
        if camera.tag != 'VCGCamera': continue

        # Translation vector
        translationVector = camera.attrib['TranslationVector']
        # Distortion
        distortion = camera.attrib['LensDistortion']
        # ViewPort
        size = camera.attrib['ViewportPx']
        # PixelSize
        pixsize = camera.attrib['PixelSizeMm']
        # CenterPx
        centerpix = camera.attrib['CenterPx']
        # Focal lenght
        focal = camera.attrib['FocalMm']
        # Rotation matrix
        rotation = camera.attrib['RotationMatrix']

        # To obtain the correct value for the focal lenght
        # we also need the pixel sizes
        focal = float(focal)
        pixsize = pixsize.split()
        focalX = focal / float(pixsize[0])
        focalY = focal / float(pixsize[1])

        # Image Dimensions
        imsize = size.split()
        imWidth = int(imsize[0])
        imHeight = int(imsize[1])
        # Pixel Center
        pixcenter = centerpix.split()
        pixCenterX = int(pixcenter[0])
        pixCenterY = int(pixcenter[1])
        # clip width of im
        if ground_file is not None:
            assert imWidth >= int(wh_ratio * imHeight), 'meshlab always height first, clip width of the image or pad with 0 (empty)'
            imWidth = int(wh_ratio * imHeight)
        pixCenterX, pixCenterY = imWidth * 0.5, imHeight * 0.5

        # Rotation
        rotation = rotation.split()
        extrinsic = []
        for i in range(0, len(rotation)):
            extrinsic.append(float(rotation[i]))

        # Position
        position = []
        translationVector = translationVector.split()
        for i in range(0,3):
            position.append(float(translationVector[i]))

        R = np.eye(3, dtype=np.float32)
        R[0, 0], R[0, 1], R[0, 2] = extrinsic[0], extrinsic[1], extrinsic[2]
        R[1, 0], R[1, 1], R[1, 2] = extrinsic[4], extrinsic[5], extrinsic[6]
        R[2, 0], R[2, 1], R[2, 2] = extrinsic[8], extrinsic[9], extrinsic[10]

        T = np.zeros((3), dtype=np.float32)
        T[0], T[1], T[2] = position[0], position[1], position[2]

        R, T = torch.from_numpy(R), torch.from_numpy(T)

        #-------------meshlab to pytorch3d------------#
        R_pytorch3d = R.T
        T_pytorch3d = T.reshape(1, 3) @ R.T

        R0 = torch.eye(3, dtype=torch.float32)
        R0[0,0], R0[2,2] = -1., -1.

        # X_pytorch3d = (X_meshlab @ P_w2c) @ R0(imHeight, imWidth)
        R_pytorch3d = R_pytorch3d @ R0
        T_pytorch3d = T_pytorch3d @ R0
        R_pytorch3d = R_pytorch3d[None]
        #-------------meshlab to pytorch3d------------#

        #-----------------pytorch3d camera------------#
        if in_ndc:
            scale = min(imWidth, imHeight) / 2.0
            focalX /= scale
            focalY /= scale
            pixCenterX = -(pixCenterX - imWidth / 2.0) / scale
            pixCenterY = -(pixCenterY - imHeight / 2.0) / scale
            print(pixCenterX, pixCenterY)
            cameras = PerspectiveCameras(
                focal_length=torch.Tensor([[focalX, focalY]]),
                principal_point=torch.Tensor([[pixCenterX, pixCenterY]]),
                R=R_pytorch3d, 
                T=T_pytorch3d,
                in_ndc=True,
            )
        else:
            cameras = PerspectiveCameras(
                focal_length=torch.Tensor([[focalX, focalY]]),
                principal_point=torch.Tensor([[pixCenterX, pixCenterY]]),
                R=R_pytorch3d, 
                T=T_pytorch3d,
                in_ndc=False,
                image_size=((imHeight, imWidth),),
            )
        #-----------------pytorch3d camera------------#

        return cameras, verts, (imHeight, imWidth), R_pytorch3d, T_pytorch3d

def fragments2depth(fragments):
    """
    Note: no depth value --> 0.0
    """
    depths = fragments.zbuf # (B, H, W, faces_per_pixel)
    depths = depths.max(dim=-1).values 
    depths = torch.nan_to_num(depths, nan=-1)
    depths[depths <= 0.] = 0.
    return depths

def prepare_renderer(device, mesh_path, xml_path, ground_file):
    obj = o3d.io.read_triangle_mesh(mesh_path)
    points = torch.from_numpy(np.asarray(obj.vertices).astype(np.float32))
    faces = torch.from_numpy(np.asarray(obj.triangles).astype(np.int32))
    rgb_colors = torch.from_numpy(np.asarray(obj.vertex_colors).astype(np.float32))

    # camera
    cameras, points, image_size, R, T = parse_xml(xml_path, points, ground_file)

    # 创建一个 3D 对象         
    textures = TexturesVertex(verts_features=rgb_colors[None])
    obj = Meshes(verts=[points], faces=[faces], textures=textures)

    # 设置光照
    # lights = PointLights(
    #     ambient_color=((0., 0., 0.), ),
    #     diffuse_color=((1.0, 1.0, 1.0),),
    #     specular_color=((0., 0., 0.), ),
    #     location=((0.0, 0.0, -3.0), ),
    #     device=device,
    #     )

    # 创建渲染器
    raster_settings = RasterizationSettings(
        image_size=image_size,
    )

    # How to Render Vertex Colored mesh?
    # https://github.com/facebookresearch/pytorch3d/issues/112
    # How can I render a 3d model "shadelessly" (with no lighting)? 
    # https://github.com/facebookresearch/pytorch3d/issues/84
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        # shader=HardPhongShader(
        #     device=device,
        #     cameras=cameras,
        #     lights=lights,
        # )
        shader=SimpleShader(device)
    )

    # to device
    renderer = renderer.to(device)
    obj = obj.to(device)
    cameras = cameras.to(device)
    R, T = R.to(device), T.to(device)

    return renderer, obj, cameras, R, T

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
        polygon = np.array(shape['points'], np.int32) # (N,2) #TODO: each area 1 point/ 2 point/ 3 point -> in total around 10 -15 oints in total
        cv2.fillPoly(img, [polygon], color=(255))

    # print(img.shape, type(img), img.max()) # (450, 800) <class 'numpy.ndarray'> 255
    return (img.astype(np.float32) / 255.).clip(0., 1.)

def blend_vis(front, behind, front_mask, white_width=5, behind_alpha=1.0):
    """
    front, behind, front_mask: (1, C, H, W)
    RGB = front[:3] * front_mask + behind[:3] * (1-front_mask)
    front_mask += pad_mask
    front += pad_mask * (white_image)
    """
    assert len(front.shape) == 4
    assert len(behind.shape) == 4
    assert len(front_mask.shape) == 4

    # white boundary
    if white_width > 0:
        # white boundary
        ori_mask = front_mask.squeeze().cpu().numpy()
        new_mask1 = cv2.erode(ori_mask, np.ones((white_width, white_width), np.uint8), iterations=1)
        new_mask2 = cv2.dilate(ori_mask, np.ones((white_width, white_width), np.uint8), iterations=1)
        pad_mask = torch.from_numpy(new_mask2 - new_mask1)[None,None,:,:].to(front.device)
        front_mask = torch.from_numpy(new_mask1)[None,None,:,:].to(front.device)
        front = front * front_mask + pad_mask*1. * (1-front_mask)
        front_mask = 1. * front_mask + pad_mask * (1-front_mask)
    
    rgb = front[:,:3] * front_mask + behind[:,:3] * (1.-front_mask)
    alpha = 1. * front_mask + behind_alpha * (1-front_mask)
    rgba = torch.cat([rgb, alpha], dim=1)
    return rgba 

if __name__ == "__main__":
    # render_top currently only works for top view, no rotation, only translation for ref view!
    # CUDA_VISIBLE_DEVICES=0 python render.py 

    colony_name = 'Devil_Island'
    # colony_name = 'Brown_Bluff'

    device = "cuda"
    root_dir = 'ATA'

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
        ref_name = 'ref'
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
        ref_name = 'ref_whole'

    xml_path_ref = f'./{root_dir}/{colony_name}/{ref_name}.xml'
    mesh_path = f'./{root_dir}/{colony_name}/data.ply'
    json_file_ref = f'./{root_dir}/{colony_name}/{ref_name}.json'
    if not os.path.exists(json_file_ref): json_file_ref=None
    save_dir = f'./{root_dir}/{colony_name}/results'

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'src_view'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'blended_src_view'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'blended_ref_view'), exist_ok=True)
    
    # top view, reference image: fixed resolution.
    if xml_path_ref is not None:
        the_renderer, the_data, the_cameras, start_R, start_T = prepare_renderer(device, mesh_path, xml_path_ref, ground_file=None)
        images, fragments = the_renderer(the_data, cameras=the_cameras, R=start_R, T=start_T)
        depth = fragments2depth(fragments) # [1, H, W]
        depth_ref = depth[0, ...] # [H, W]
        to_pil_image(images.permute(0, 3, 1, 2)[0].cpu()).save(f'{save_dir}/rgba_ref.png')
        rgba_ref = images.permute(0, 3, 1, 2).clone() # (1, C, H, W)
    if json_file_ref is not None:
        rgba_ref_mask = sam_post(json_file_ref) 
        rgba_ref_mask = torch.from_numpy(rgba_ref_mask[None,None,:,:]).to(device)

    front = torch.zeros_like(rgba_ref[:,:3])
    front_mask = torch.zeros_like(rgba_ref[:,3:])
    for img_name in img_names:
        logger.info(f'-------{img_name}: start------------')
        
        # find the image
        for _ext in ['png', 'jpg', 'JPG', 'jpeg', 'PNG']:
            ground_file = f'./{root_dir}/{colony_name}/{img_name}.{_ext}'
            if os.path.exists(ground_file):
                break
        assert os.path.exists(ground_file), f'{ground_file} does not exist'
        # find mask & camera pose file
        json_file = f'./{root_dir}/{colony_name}/{img_name}.json' # segmentation mask
        xml_path = f'./{root_dir}/{colony_name}/{img_name}.xml' # camera pose

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
        plt.imsave(f'{save_dir}/rendered_image.png', images_np)
        # plt.imsave(f'{save_dir}/rendered_image_alpha.png', (images[0, ..., 3:]).cpu().numpy())

        dmi, dma = depth[depth>0].min(), depth[depth>0].max()
        depth_gray = ((depth-dmi) / (dma-dmi + 1e-8)).clip(0.,1.)
        plt.imsave(f'{save_dir}/rendered_depth.png', 1. - depth_gray[0, ...].cpu().numpy())
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
        # src view
        _ground = ground_Img.permute(0, 3, 1, 2).clone()
        _sam = torch.from_numpy(sam_mask[None,:,:,None].astype(np.float32))
        _src_view = blend_vis(_ground, _ground, _sam)
        to_pil_image(_src_view[0, ...].cpu()).save(f'{save_dir}/src_view/{img_name}.png')
        # mask sam
        sam_mask &= (alpha_mask > 0) & (depth_np > 0)
        # sam_dmi, sam_dma = np.quantile(depth_np[sam_mask], 0.1), np.quantile(depth_np[sam_mask], 0.9)
        # sam_mask &= (depth_np >= sam_dmi) & (depth_np <= sam_dma)
        # _m = copy.deepcopy(sam_mask)
        # sam_mask[:-1,:-1] &= _m[1:,:-1] & _m[:-1,1:] & _m[1:,1:]
        # sam_mask[1:,1:] &= _m[1:,:-1] & _m[:-1,1:] & _m[:-1,:-1]
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
        to_pil_image(blended_src_view[0, ...].cpu()).save(f'{save_dir}/blended_src_view/{img_name}.png')
        # blended - ref view
        sampled_rgba_src = render_top(rgba_src, pose_ref, pose_src, intrinsics_ref, intrinsics_src, depth_ref, depth_src=depth_src)
        if json_file_ref is not None:
            sampled_rgba_src *= rgba_ref_mask # w ref mask
        blended_ref_view = blend_vis(sampled_rgba_src, rgba_ref, sampled_rgba_src[:,3:], white_width=3)
        to_pil_image(blended_ref_view[0,:,:,:].cpu()).save(f'{save_dir}/blended_ref_view/{img_name}.png')
        # to_pil_image(blended_ref_view[0,:,render_h//4:-render_h//4,render_w//4:-render_w//4].cpu()).save(f'{save_dir}/blended_ref_view/{img_name}_zoom.png')

        # RGB += (1-M) * (new_RGB * new_M)
        # M += (1-M) * new_M
        front += (1-front_mask) * sampled_rgba_src[:,:3]
        front_mask += (1-front_mask) * sampled_rgba_src[:,3:]

        logger.info(f'-------{json_file}: end------------')

    # blended - ref view (total)
    logger.info(f'-------blended - ref view (total)------------')
    final = blend_vis(front, rgba_ref, front_mask, white_width=3)
    to_pil_image(final[0,...].cpu()).save(f'{save_dir}/blended.png')
    to_pil_image(front_mask[0,...].cpu()).save(f'{save_dir}/pred_mask.png')