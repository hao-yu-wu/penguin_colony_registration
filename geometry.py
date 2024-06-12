from kornia.geometry import depth
import torch 
import numpy as np
import kornia
import cv2
from splatting import splatting_function
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

NORMALIZE_SOBEL = False  # normalize sobel gradient

def dilate(image, kernel_shape=(31,31)):
    """
    Dilate an image using a specific structuring element.
    
    Args:
        image (torch.Tensor): The input image.
        kernel (torch.Tensor): The structuring element.
        
    Returns:
        torch.Tensor: The dilated image.
    """
    assert len(image.shape) == 4

    padding = tuple(side // 2 for side in kernel_shape)

    output = F.max_pool2d(image, kernel_size=kernel_shape, stride=1, padding=padding)

    return output

def pose2xyz():
    '''
    Args:
        poses: [B, 4, 4], cam2world, P @ X_cam = X_world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''
    poses = torch.rand((1, 4, 4), dtype=torch.float32)
    H, W = 512, 512

    device = poses.device
    fx, fy, cx, cy = 200, 200, H/2, W/2

    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device), indexing='ij')
    i = i.t().reshape([1, H*W]) + 0.5 # (1, 512*512)
    j = j.t().reshape([1, H*W]) + 0.5 # (1, 512*512)

    zs = - torch.ones_like(i) # (1, 512*512)
    xs = - (i - cx) / fx * zs # (1, 512*512)
    ys = (j - cy) / fy * zs # (1, 512*512)
    directions = torch.stack((xs, ys, zs), dim=-1) # (1, 512*512, 3)
    # directions = safe_normalize(directions)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (1, 512*512, 3)

    rays_o = poses[..., :3, 3] # [1, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # (1, 512*512, 3)

    N = rays_o.shape[0] # N=1
    z = 0.5
    xyzs = rays_o + rays_d * z # (1, 512*512, 3)

def safe_inverse(A):
    # (4,4) -> (4,4)
    # torch.inverse(A)
    R = A[:3, :3] # 3,3
    T = A[:3, 3:] # 3,1
    R_prime = R.T
    T_prime = - R.T @ T
    A[:3, :3] = R_prime
    A[:3, 3:] = T_prime
    return A

def sobel_alpha(x, mode='sobel', beta=10.0, detach=True):
    # https://github.com/google-research/google-research/blob/06dbe65406762e7fe449c4610bdfd8897c4ebc33/infinite_nature_zero/models/render_model.py#L156
    sobel_grad = kornia.filters.spatial_gradient(x, mode=mode, normalized=NORMALIZE_SOBEL)
    sobel_mag = torch.sqrt(sobel_grad[:, :, 0, Ellipsis]**2 + sobel_grad[:, :, 1, Ellipsis]**2)
    if detach:
        alpha = torch.exp(-1.0 * beta * sobel_mag).detach()
    else:
        alpha = torch.exp(-1.0 * beta * sobel_mag)
    return alpha

def render_forward_splat(src_imgs, src_depths, pose_ref, pose_src, intrinsics_ref, intrinsics_src, alpha_threshold=0.3, mask_threshold=0.9):
    """3D render the image to the next viewpoint.
    Args:
      -- tensors --
      src_imgs: source images (B, H, W, C) = (1, 128, 128, 4)
      src_depths: source depth maps (B, H, W) 
      r_cam: relative camera rotation (B, 3, 3)
      t_cam: relative camera translation (B, 3)
      k_src: source intrinsic matrix (B, 3, 3)
      k_dst: destination intrinsic matrix (B, 3, 3)
    Returns:
      warp_feature: the rendered RGB feature map [B, 4, 128, 128]
      warp_disp: the rendered disparity [B, 1, 128, 128]
      warp_mask: the rendered mask [B, 1, 128, 128]
    """
    pose_ref = pose_ref.float()
    pose_src = pose_src.float()
    src_imgs = src_imgs.float()
    src_depths = src_depths.float()

    src_depths_alpha = sobel_alpha(src_depths[:,None,:,:]) # B,1,H,W
    src_imgs[:, :, :, 3:] = src_imgs[:, :, :, 3:] * src_depths_alpha.permute(0, 2, 3, 1)

    batch_size = src_imgs.shape[0]
    assert batch_size == 1

    device = src_depths.device
    _K_ref = intrinsics_ref.astype(np.float32) # np
    K_ref = np.zeros((3, 3), dtype=np.float32)
    K_ref[0, 0], K_ref[1, 1], K_ref[0, 2], K_ref[1, 2], K_ref[2, 2] = _K_ref[0], _K_ref[1], _K_ref[2], _K_ref[3], 1
    k_dst = torch.from_numpy(K_ref).to(device).unsqueeze(0)
    RT_ref = pose_ref.squeeze()
    RT_ref[:3, 1:3] *= -1

    _K_src = intrinsics_src.astype(np.float32) # np
    K_src = np.zeros((3, 3), dtype=np.float32)
    K_src[0, 0], K_src[1, 1], K_src[0, 2], K_src[1, 2], K_src[2, 2] = _K_src[0], _K_src[1], _K_src[2], _K_src[3], 1
    k_src = torch.from_numpy(K_src).to(device).unsqueeze(0)
    RT_src = pose_src.squeeze()
    RT_src[:3, 1:3] *= -1

    # relative pose [cam2world]: src -> ref
    r_pose = torch.matmul(safe_inverse(RT_ref), RT_src) # (4, 4)
    # or 
    # RT_ref = safe_inverse(RT_ref) # (4, 4)
    # RT_src = safe_inverse(RT_src) # (4, 4)
    # r_pose = torch.matmul(RT_ref, safe_inverse(RT_src)) # (4, 4)

    r_pose = r_pose.unsqueeze(0)
    rot, t = r_pose[:, :3, :3], r_pose[:, :3, 3]

    k_src_inv = k_src.inverse()

    x = np.arange(src_imgs[0].shape[1])
    y = np.arange(src_imgs[0].shape[0])
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)
    coord = coord.astype(np.float32)
    coord = torch.as_tensor(coord, dtype=k_src.dtype, device=k_src.device)
    coord = coord[None, Ellipsis, None].repeat(batch_size, 1, 1, 1, 1)

    depth = src_depths[:, :, :, None, None] # (B, H, W, 1, 1)

    # from reference to target viewpoint
    pts_3d_ref = depth * k_src_inv[:, None, None, Ellipsis] @ coord
    pts_3d_tgt = rot[:, None, None, Ellipsis] @ pts_3d_ref + t[:, None, None, :, None]
    points = k_dst[:, None, None, Ellipsis] @ pts_3d_tgt
    points = points.squeeze(-1)

    new_z = points[:, :, :, [2]].clone().permute(0, 3, 1, 2)  # b,1,h,w
    points = points / torch.clamp(points[:, :, :, [2]], 1e-8, None) # numerical stability for depth, which > 0

    src_ims_ = src_imgs.permute(0, 3, 1, 2) # b,c,h,w
    num_channels = src_ims_.shape[1]

    flow = points - coord.squeeze(-1)
    flow = flow.permute(0, 3, 1, 2)[:, :2, Ellipsis]

    importance = 1. / (new_z + 1e-8) # TODO: added by me for numerical stability
    importance_min = importance.amin((1, 2, 3), keepdim=True)
    importance_max = importance.amax((1, 2, 3), keepdim=True)
    # TODO
    weights = (importance - importance_min) / (importance_max - importance_min + 1e-6) * 20 - 10
    # src_mask_ = 1. * (src_depths > 0).unsqueeze(1)
    src_mask_ = torch.ones_like(new_z) # b,1,h,w

    input_data = torch.cat([src_ims_, (1. / (new_z + 1e-8)), src_mask_], 1) # TODO: added by me for numerical stability

    output_data = splatting_function('softmax', input_data, flow, weights.detach())

    warp_rgba = output_data[:, 0:num_channels, Ellipsis]
    warp_disp = output_data[:, num_channels:num_channels + 1, Ellipsis]
    warp_depth = 1. / (warp_disp + 1e-8) # TODO: added by me for numerical stability
    warp_mask = output_data[:, num_channels + 1:num_channels + 2, Ellipsis]
    
    disocc_mask = (warp_rgba[:, 3:4] > alpha_threshold) * (warp_mask > mask_threshold)
    disocc_mask = disocc_mask.detach()
    warp_rgba_masked = warp_rgba * disocc_mask # alpha channel: 0..1
    warp_depth_masked = warp_depth * disocc_mask

    warp_rgba_masked = warp_rgba_masked.clamp(0., 1.)
    return warp_rgba_masked, warp_depth_masked, disocc_mask

def render_top(rgba_src, pose_ref, pose_src, intrinsics_ref, intrinsics_src, depth_ref, depth_src=None):
    """
    rgba_src: 1,4,H_src,W_src
    depth_ref: H, W, tensor
    intrinsics_ref: 4,
    pose_ref: 4,4
        pose is world2cam: X_cam = P @ X_world
            (4, 4) @ (4, N) -> (4, N)
        meshlab:
            X_c.T (3,1) = R @ (X_w.T + T) = R @ X_w.T + R @ T
            P --> R T
                0 1

    return B,4,H,W
    """
    rgba_src = rgba_src.float()
    depth_ref = depth_ref.float()
    if depth_src is not None:
        depth_src = depth_src.float()
    H_src, W_src = rgba_src.shape[2], rgba_src.shape[3] # 1,4,H_src,W_src

    device = depth_ref.device

    _K_ref = intrinsics_ref.astype(np.float32) # np
    K_ref = np.zeros((3, 3), dtype=np.float32)
    K_ref[0, 0], K_ref[1, 1], K_ref[0, 2], K_ref[1, 2], K_ref[2, 2] = _K_ref[0], _K_ref[1], _K_ref[2], _K_ref[3], 1
    K_ref = torch.from_numpy(K_ref).to(device)
    RT_ref = torch.from_numpy(pose_ref).to(device) # (4, 4)

    _K_src = intrinsics_src.astype(np.float32) # np
    K_src = np.zeros((3, 3), dtype=np.float32)
    K_src[0, 0], K_src[1, 1], K_src[0, 2], K_src[1, 2], K_src[2, 2] = _K_src[0], _K_src[1], _K_src[2], _K_src[3], 1
    K_src = torch.from_numpy(K_src).to(device)
    RT_src = torch.from_numpy(pose_src).to(device) # (4, 4)

    if depth_src is not None:
        geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(
            depth_ref=depth_ref.squeeze(), intrinsics_ref=K_ref, extrinsics_ref=RT_ref,
            depth_src=depth_src.squeeze(), intrinsics_src=K_src, extrinsics_src=RT_src,
            H_src=H_src, W_src=W_src,
            filter_dist=1e7, filter_diff=10.)#0.1)#0.0002)
    else:
        x2d_src, y2d_src = reproject_with_depth_simple(
            depth_ref=depth_ref.squeeze(), intrinsics_ref=K_ref, extrinsics_ref=RT_ref,
            intrinsics_src=K_src, extrinsics_src=RT_src)
    sampled_rgba_src = kornia.geometry.transform.remap(
            rgba_src, x2d_src, y2d_src, mode='bicubic', padding_mode='zeros', align_corners=False) # 1,4,64,64
    # x2d <-- intrinsics_ref[0,:] <-- width
    # y2d <-- extrinsics_ref[1,:] <-- height
    sampled_alpha_src = (x2d_src < W_src) & (x2d_src >= 0) & (y2d_src < H_src) & (y2d_src >= 0)
    sampled_alpha_src = sampled_alpha_src.detach()
    if depth_src is not None:
        sampled_alpha_src *= geo_mask[None, ...].detach()
    sampled_rgba_src[:, 3] *= 1.*sampled_alpha_src
    # depth_reprojected = depth_reprojected[None, :,:]
    sampled_rgba_src = sampled_rgba_src.clamp(0., 1.)
    return sampled_rgba_src

def render_via_sample(rgba_src, pose_ref, pose_src, intrinsics_ref, intrinsics_src, depth_ref, depth_src=None):
    """
    return B,4,H,W
    """
    pose_ref = pose_ref.float()
    pose_src = pose_src.float()
    rgba_src = rgba_src.float()
    depth_ref = depth_ref.float()
    if depth_src is not None:
        depth_src = depth_src.float()
    H_src, W_src = rgba_src.shape[2], rgba_src.shape[3] # 1,4,H_src,W_src

    device = depth_ref.device

    _K_ref = intrinsics_ref.astype(np.float32) # np
    K_ref = np.zeros((3, 3), dtype=np.float32)
    K_ref[0, 0], K_ref[1, 1], K_ref[0, 2], K_ref[1, 2], K_ref[2, 2] = _K_ref[0], _K_ref[1], _K_ref[2], _K_ref[3], 1
    K_ref = torch.from_numpy(K_ref).to(device)
    RT_ref = pose_ref.squeeze()
    RT_ref[:3, 1:3] *= -1
    RT_ref = safe_inverse(RT_ref) # (4, 4)

    _K_src = intrinsics_src.astype(np.float32) # np
    K_src = np.zeros((3, 3), dtype=np.float32)
    K_src[0, 0], K_src[1, 1], K_src[0, 2], K_src[1, 2], K_src[2, 2] = _K_src[0], _K_src[1], _K_src[2], _K_src[3], 1
    K_src = torch.from_numpy(K_src).to(device)
    RT_src = pose_src.squeeze()
    RT_src[:3, 1:3] *= -1
    RT_src = safe_inverse(RT_src) # (4, 4)

    if depth_src is not None:
        geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(
            depth_ref=depth_ref.squeeze(), intrinsics_ref=K_ref, extrinsics_ref=RT_ref,
            depth_src=depth_src.squeeze(), intrinsics_src=K_src, extrinsics_src=RT_src,
            filter_dist=10, filter_diff=1)
    else:
        x2d_src, y2d_src = reproject_with_depth_simple(
            depth_ref=depth_ref.squeeze(), intrinsics_ref=K_ref, extrinsics_ref=RT_ref,
            intrinsics_src=K_src, extrinsics_src=RT_src)
    sampled_rgba_src = kornia.geometry.transform.remap(
            rgba_src, x2d_src, y2d_src, mode='bicubic', padding_mode='zeros', align_corners=False) # 1,4,64,64
    sampled_alpha_src = (x2d_src < W_src) & (x2d_src >= 0) & (y2d_src < H_src) & (y2d_src >= 0)
    sampled_alpha_src = sampled_alpha_src.detach()
    if depth_src is not None:
        sampled_alpha_src *= geo_mask[None, ...].detach()
    sampled_rgba_src[:, 3] *= 1.*sampled_alpha_src
    # depth_reprojected = depth_reprojected[None, :,:]
    sampled_rgba_src = sampled_rgba_src.clamp(0., 1.)
    return sampled_rgba_src

def reproject_with_depth_simple(depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src, eps=1e-4):
    height, width = depth_ref.shape[0], depth_ref.shape[1]
    dtype = depth_ref.dtype
    device = depth_ref.device
    
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = torch.meshgrid(torch.arange(0, width), torch.arange(0, height), indexing='xy')
    x_ref, y_ref = x_ref.to(dtype).to(device), y_ref.to(dtype).to(device)
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = torch.matmul(torch.inverse(intrinsics_ref), 
                           torch.vstack((x_ref, y_ref, torch.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    # (4, 4) @ (4, N) -> (4, N)
    xyz_src = torch.matmul(torch.matmul(extrinsics_src, safe_inverse(extrinsics_ref)),
                           torch.vstack((xyz_ref, torch.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)
    # depth = 0 is useless
    K_xyz_src[2][torch.abs(K_xyz_src[2]) < eps] = eps
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).to(dtype)[None, ...]
    y_src = xy_src[1].reshape([height, width]).to(dtype)[None, ...]

    return x_src, y_src

def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src, eps=1e-4):
    height, width = depth_ref.shape[0], depth_ref.shape[1]
    dtype = depth_ref.dtype
    device = depth_ref.device
    
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = torch.meshgrid(torch.arange(0, width), torch.arange(0, height), indexing='xy')
    x_ref, y_ref = x_ref.to(dtype).to(device), y_ref.to(dtype).to(device)
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = torch.matmul(torch.inverse(intrinsics_ref), 
                           torch.vstack((x_ref, y_ref, torch.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = torch.matmul(torch.matmul(extrinsics_src, safe_inverse(extrinsics_ref)),
                           torch.vstack((xyz_ref, torch.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)
    # depth = 0 is useless
    K_xyz_src[2][torch.abs(K_xyz_src[2]) < eps] = eps
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).to(dtype)[None, ...]
    y_src = xy_src[1].reshape([height, width]).to(dtype)[None, ...]
    sampled_depth_src = kornia.geometry.transform.remap(depth_src[None, None, :, :], x_src, y_src, 
                                                    mode='bilinear', padding_mode='zeros', align_corners=False).squeeze() # (1, 1, H, W) -> (H, W)

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = torch.matmul(torch.inverse(intrinsics_src),
                           torch.vstack((xy_src, torch.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = torch.matmul(torch.matmul(extrinsics_ref, safe_inverse(extrinsics_src)),
                                   torch.vstack((xyz_src, torch.ones_like(x_ref))))[:3]
    # depth = 0 is useless anyway
    xyz_reprojected[2][torch.abs(xyz_reprojected[2]) < eps] = eps
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).to(dtype)
    K_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).to(dtype)
    y_reprojected = xy_reprojected[1].reshape([height, width]).to(dtype)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src

def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src, H_src=None, W_src=None, filter_dist=1, filter_diff=0.01, eps=1e-4):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = torch.meshgrid(torch.arange(0, width), torch.arange(0, height), indexing='xy')
    x_ref, y_ref = x_ref.to(depth_ref.dtype).to(depth_ref.device), y_ref.to(depth_ref.dtype).to(depth_ref.device)
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(
        depth_ref, intrinsics_ref, extrinsics_ref,
        depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = torch.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = torch.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / (depth_ref + eps)

    mask = torch.logical_and(dist < filter_dist, relative_depth_diff < filter_diff)

    # d_mask = (x2d_src < W_src) & (x2d_src >= 0) & (y2d_src < H_src) & (y2d_src >= 0)
    # d_mask = d_mask.squeeze()
    # dist_vis = (dist - dist[d_mask].min()) / (dist[d_mask].max() - dist[d_mask].min())
    # dist_vis[~d_mask] = 1.
    # diff_vis = (relative_depth_diff - relative_depth_diff[d_mask].min()) / (relative_depth_diff[d_mask].max() - relative_depth_diff[d_mask].min())
    # diff_vis[~d_mask] = 1.
    # dist_vis = 1. * (dist_vis.clip(0.,1.) < 0.15)
    # diff_vis = 1. * (diff_vis.clip(0.,1.) < 0.01)
    # print(dist[dist_vis > 0.5].max(), relative_depth_diff[diff_vis > 0.5].max())
    # # 1599.1385 0.0002
    # to_pil_image(dist_vis.detach().cpu()).save('dist.png')
    # to_pil_image(diff_vis.detach().cpu()).save('diff.png')
    # mask = torch.logical_and(dist_vis, diff_vis)

    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src

def _render_via_sample(rgba_src, pose_ref, pose_src, intrinsics_ref, intrinsics_src, depth_ref, depth_src, H, W):
    """
    return B,4,H,W
    """
    pose_ref = pose_ref.float()
    pose_src = pose_src.float()
    rgba_src = rgba_src.float()
    depth_ref = depth_ref.float()
    depth_src = depth_src.float()

    device = depth_ref.device
    _K_ref = intrinsics_ref.astype(np.float32) # np
    K_ref = np.zeros((3, 3), dtype=np.float32)
    K_ref[0, 0], K_ref[1, 1], K_ref[0, 2], K_ref[1, 2], K_ref[2, 2] = _K_ref[0], _K_ref[1], _K_ref[2], _K_ref[3], 1
    # K_ref = torch.from_numpy(K_ref).to(device)
    RT_ref = pose_ref.squeeze()
    RT_ref[:3, 1:3] *= -1
    RT_ref = safe_inverse(RT_ref) # (4, 4)

    _K_src = intrinsics_src.astype(np.float32) # np
    K_src = np.zeros((3, 3), dtype=np.float32)
    K_src[0, 0], K_src[1, 1], K_src[0, 2], K_src[1, 2], K_src[2, 2] = _K_src[0], _K_src[1], _K_src[2], _K_src[3], 1
    # K_src = torch.from_numpy(K_src).to(device)
    RT_src = pose_src.squeeze()
    RT_src[:3, 1:3] *= -1
    RT_src = safe_inverse(RT_src) # (4, 4)

    geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(
        depth_ref=depth_ref.squeeze().cpu().numpy(), intrinsics_ref=K_ref, extrinsics_ref=RT_ref.cpu().numpy(),
        depth_src=depth_src.squeeze().cpu().numpy(), intrinsics_src=K_src, extrinsics_src=RT_src.cpu().numpy(),
        filter_dist=4, filter_diff=0.1)
    img_src = rgba_src.permute(0,2,3,1).squeeze().cpu().numpy() # B,4,H,W -> B,H,W,4 -> H,W,4
    sampled_img_src = cv2.remap(img_src, x2d_src, y2d_src, interpolation=cv2.INTER_CUBIC) # (H,W,4) -> (?)
    sampled_alpha_src = (x2d_src < W) & (x2d_src >= 0) & (y2d_src < H) & (y2d_src >= 0)
    sampled_img_src *= 1.*sampled_alpha_src[..., None]
    sampled_img_src = torch.from_numpy(sampled_img_src).permute(2,0,1)[None,...].to(device)
    depth_reprojected = torch.from_numpy(depth_reprojected)[None,...].to(device)
    return sampled_img_src, depth_reprojected

def _reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    # extrinsics_src is world2cam: X_cam = P @ X_world
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space: ref -> src
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src

def _check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src, filter_dist=1, filter_diff=0.01):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < filter_dist, relative_depth_diff < filter_diff)
    # mask = np.logical_and(dist < 1, relative_depth_diff < 0.01)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src

if __name__ == '__main__':
    pose2xyz()