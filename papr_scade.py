import numpy as np
import argparse
import json
import torch
import cv2
import shutil
import os
import yaml
import subprocess
import torchvision
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity

from models import get_model 
from data import create_random_subsets, load_scene_scannet
from prior_utils import get_ray_batch_from_one_image_hypothesis_idx
from papr_helpers import compute_space_carving_loss_papr
from papr_scade_utils import config_parser, dict_to_namespace, load_checkpoint, \
    to8b, to16b,  img2mse, mse2psnr, compute_rmse, get_learning_rate, update_learning_rate, get_rays, MeanTracker
from tqdm import tqdm, trange
from lpips import LPIPS 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### RENDERING ###
def optimize_camera_embedding(image, pose, H, W, model, config, intrinsic, args, render_kwargs_test):
    render_kwargs_test["embedded_cam"] = torch.zeros(args.input_ch_cam, requires_grad=True).to(device)
    optimizer = torch.optim.Adam(params=(render_kwargs_test["embedded_cam"],), lr=5e-1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3, verbose=True)
    half_W = W
    print(" - Optimize camera embedding")
    max_psnr = 0
    best_embedded_cam = torch.zeros(args.input_ch_cam).to(device)
    # make batches
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, half_W - 1, half_W), indexing='ij'), -1)  # (H, W, 2)
    coords = torch.reshape(coords, [-1, 2]).long()
    assert(coords[:, 1].max() < half_W)
    batches = create_random_subsets(range(len(coords)), 2 * args.N_rand, device=device)
    # make rays
    rays_o, rays_d = get_rays(H, half_W, intrinsic, pose)  # (H, W, 3), (H, W, 3)
    start_time = time.time()
    for i in range(100):
        sum_img_loss = torch.zeros(1)
        optimizer.zero_grad()
        for b in batches:
            curr_coords = coords[b]
            curr_rays_o = rays_o[curr_coords[:, 0], curr_coords[:, 1]]  # (N_rand, 3)
            curr_rays_d = rays_d[curr_coords[:, 0], curr_coords[:, 1]]  # (N_rand, 3)
            target_s = image[curr_coords[:, 0], curr_coords[:, 1]]
            batch_rays = torch.stack([curr_rays_o, curr_rays_d], 0)
            rgb, _, _, _ = render(H, half_W, None, model, config, chunk=args.chunk, rays=batch_rays, verbose=i < 10, **render_kwargs_test)
            img_loss = img2mse(rgb, target_s)
            img_loss.backward()
            sum_img_loss += img_loss
        optimizer.step()
        psnr = mse2psnr(sum_img_loss / len(batches))
        lr_scheduler.step(psnr)
        if psnr > max_psnr:
            max_psnr = psnr
            best_embedded_cam = render_kwargs_test["embedded_cam"].detach().clone()
            print("Step {}: PSNR: {} ({:.2f}min)".format(i, psnr, (time.time() - start_time) / 60))
    render_kwargs_test["embedded_cam"] = best_embedded_cam

def render_images_with_metrics(model, count, indices, images, depths, valid_depths, poses, H, W, intrinsics, lpips_alex, args, config, render_kwargs_test, \
    embedcam_fn=None, with_test_time_optimization=False):
    far = render_kwargs_test['far']

    if count is None:
        # take all images in order
        count = len(indices)
        img_i = indices
    else:
        # take random images
        img_i = np.random.choice(indices, size=count, replace=False)

    rgbs_res = torch.empty(count, 3, H, W)
    target_rgbs_res = torch.empty(count, 3, H, W)
    depths_res = torch.empty(count, 1, H, W)
    target_depths_res = torch.empty(count, 1, H, W)
    target_valid_depths_res = torch.empty(count, 1, H, W, dtype=bool)
    
    mean_metrics = MeanTracker()
    mean_depth_metrics = MeanTracker() # track separately since they are not always available
    for n, img_idx in enumerate(img_i):
        print("Render image {}/{}".format(n + 1, count), end="")
        target = images[img_idx]
        target_depth = depths[img_idx]
        target_valid_depth = valid_depths[img_idx]
        pose = poses[img_idx, :3,:4]
        intrinsic = intrinsics[img_idx, :]

        if args.input_ch_cam > 0:
            if embedcam_fn is None:
                # use zero embedding at test time or optimize for the latent code
                render_kwargs_test["embedded_cam"] = torch.zeros((args.input_ch_cam), device=device)
                if with_test_time_optimization:
                    optimize_camera_embedding(target, pose, H, W, intrinsic, args, render_kwargs_test)
                    result_dir = os.path.join(args.ckpt_dir, args.expname, "test_latent_codes_" + args.scene_id)
                    os.makedirs(result_dir, exist_ok=True)
                    np.savetxt(os.path.join(result_dir, str(img_idx) + ".txt"), render_kwargs_test["embedded_cam"].cpu().numpy())
            else:
                render_kwargs_test["embedded_cam"] = embedcam_fn[img_idx]
        
        with torch.no_grad():
            rgb, depth = render(H, W, intrinsic, model, config, chunk=(args.chunk // 2), c2w=pose, **render_kwargs_test)
            
            # compute depth rmse
            depth_rmse = compute_rmse(depth[target_valid_depth], target_depth[:, :, 0][target_valid_depth])
            if not torch.isnan(depth_rmse):
                depth_metrics = {"depth_rmse" : depth_rmse.item()}
                mean_depth_metrics.add(depth_metrics)

            ### Fit LSTSQ for white balancing
            rgb_reshape = rgb.view(1, -1, 3)
            target_reshape = target.view(1, -1, 3)

            ## No intercept          
            # X = torch.linalg.lstsq(rgb_reshape, target_reshape).solution
            # rgb_reshape = rgb_reshape @ X
            # rgb_reshape = rgb_reshape.view(rgb.shape)
            # rgb = rgb_reshape
            
            # compute color metrics
            img_loss = img2mse(rgb, target)
            psnr = mse2psnr(img_loss)
            print("PSNR: {}".format(psnr))
            rgb = torch.clamp(rgb, 0, 1)
            ssim = structural_similarity(rgb.cpu().numpy(), target.cpu().numpy(), data_range=1., channel_axis=-1)
            lpips = lpips_alex(rgb.permute(2, 0, 1).unsqueeze(0), target.permute(2, 0, 1).unsqueeze(0), normalize=True)[0]
            
            # store result
            rgbs_res[n] = rgb.clamp(0., 1.).permute(2, 0, 1).cpu()
            target_rgbs_res[n] = target.permute(2, 0, 1).cpu()
            depths_res[n] = (depth / far).unsqueeze(0).cpu()
            target_depths_res[n] = (target_depth[:, :, 0] / far).unsqueeze(0).cpu()
            target_valid_depths_res[n] = target_valid_depth.unsqueeze(0).cpu()
            metrics = {"img_loss" : img_loss.item(), "psnr" : psnr.item(), "ssim" : ssim, "lpips" : lpips[0, 0, 0],}

            mean_metrics.add(metrics)
    
    res = { "rgbs" :  rgbs_res, "target_rgbs" : target_rgbs_res, "depths" : depths_res, "target_depths" : target_depths_res, \
        "target_valid_depths" : target_valid_depths_res}
    all_mean_metrics = MeanTracker()
    all_mean_metrics.add({**mean_metrics.as_dict(), **mean_depth_metrics.as_dict()})
    return all_mean_metrics, res

def write_images_with_metrics(images, mean_metrics, far, args, with_test_time_optimization=False):
    result_dir = os.path.join(args.ckpt_dir, args.expname, "test_images_" + ("with_optimization_" if with_test_time_optimization else "") + args.scene_id)
    os.makedirs(result_dir, exist_ok=True)
    for n, (rgb, depth) in enumerate(zip(images["rgbs"].permute(0, 2, 3, 1).cpu().numpy(), \
            images["depths"].permute(0, 2, 3, 1).cpu().numpy())):

        # write rgb
        cv2.imwrite(os.path.join(result_dir, str(n) + "_rgb" + ".jpg"), cv2.cvtColor(to8b(rgb), cv2.COLOR_RGB2BGR))
        # write depth
        cv2.imwrite(os.path.join(result_dir, str(n) + "_d" + ".png"), to16b(depth))

    with open(os.path.join(result_dir, 'metrics.txt'), 'w') as f:
        mean_metrics.print(f)
    mean_metrics.print()

def render_video(poses, H, W, model, config, intrinsics, filename, args, render_kwargs_test, fps=25):
    video_dir = os.path.join(args.ckpt_dir, args.expname, 'video_' + filename)
    if os.path.exists(video_dir):
        shutil.rmtree(video_dir)
    os.makedirs(video_dir, exist_ok=True)
    depth_scale = render_kwargs_test["far"]
    max_depth_in_video = 0
    for img_idx in range(0, len(poses), 3):
    # for img_idx in range(200):
        pose = poses[img_idx, :3,:4]
        intrinsic = intrinsics[img_idx, :]
        with torch.no_grad():
            if args.input_ch_cam > 0:
                render_kwargs_test["embedded_cam"] = torch.zeros((args.input_ch_cam), device=device)
            # render video in 16:9 with one third rgb, one third depth and one third depth standard deviation
            rgb, depth = render(H, W, intrinsic, model, config, chunk=(args.chunk // 2), c2w=pose, with_5_9=True, **render_kwargs_test)
            rgb_cpu_numpy_8b = to8b(rgb.cpu().numpy())
            video_frame = cv2.cvtColor(rgb_cpu_numpy_8b, cv2.COLOR_RGB2BGR)
            max_depth_in_video = max(max_depth_in_video, depth.max())
            depth_frame = cv2.applyColorMap(to8b((depth / depth_scale).cpu().numpy()), cv2.COLORMAP_TURBO)
            video_frame = np.concatenate((video_frame, depth_frame), 1)
            # depth_var = ((extras['z_vals'] - extras['depth_map'].unsqueeze(-1)).pow(2) * extras['weights']).sum(-1)
            # depth_std = depth_var.clamp(0., 1.).sqrt()
            # video_frame = np.concatenate((video_frame, cv2.applyColorMap(to8b(depth_std.cpu().numpy()), cv2.COLORMAP_VIRIDIS)), 1)
            video_frame = np.concatenate((video_frame, cv2.COLORMAP_VIRIDIS), 1)
            cv2.imwrite(os.path.join(video_dir, str(img_idx) + '.jpg'), video_frame)

    video_file = os.path.join(args.ckpt_dir, args.expname, filename + '.mp4')
    subprocess.call(["ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(video_dir, "%d.jpg"), "-c:v", "libx264", "-profile:v", "high", "-crf", str(fps), video_file])
    print("Maximal depth in video: {}".format(max_depth_in_video))

def render_papr(ray_batch, model, use_viewdirs, config, c2w=None, **kwargs):

    ### DELETE ###
    #print('5 - Render PAPR')

    HW = ray_batch.shape[0]

    N = 1
    W = 156
    H = int(HW/W)

    #print(N,H,W)
    

    # Get information from the batch
    N_rays = ray_batch.shape[0]
    rayo, rayd = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each

    # get the feature map and attn
    # copied from PAPR test.py test_step line 49
    # H, W = rayd.shape
    num_pts, _ = model.points.shape

    rayo = rayo.to(device)
    rayd = rayd.to(device)
    #c2w = c2w.to(device)

    topk = min([num_pts, model.select_k])
    selected_points = torch.zeros(N, H, W, topk, 3)

    bkg_seq_len_attn = 0
    tx_opt = config.models.transformer
    feat_dim = tx_opt.embed.d_ff_out if tx_opt.embed.share_embed else tx_opt.embed.value.d_ff_out
    if model.bkg_feats is not None:
        bkg_seq_len_attn = model.bkg_feats.shape[0]
    feature_map = torch.zeros(N, H, W, 1, feat_dim).to(device)
    attn = torch.zeros(N, H, W, topk + bkg_seq_len_attn, 1).to(device)

    #print(rayd.shape)
    rayd = rayd.reshape([N,H,W,3])
    rayo = rayo[0].unsqueeze(0)
    #print(rayd.shape)

    #print(rayo.shape)
    
    with torch.no_grad():
        for height_start in range(0, H, config.test.max_height):
            for width_start in range(0, W, config.test.max_width):
                height_end = min(height_start + config.test.max_height, H)
                width_end = min(width_start + config.test.max_width, W)

                #### TODO: STEP
                feature_map[:, height_start:height_end, width_start:width_end, :, :], \
                attn[:, height_start:height_end, width_start:width_end, :, :] = model.evaluate(rayo, rayd[:, height_start:height_end, width_start:width_end], c2w, step=-1)

                selected_points[:, height_start:height_end, width_start:width_end, :, :] = model.selected_points

        if config.models.use_renderer:
            foreground_rgb = model.renderer(feature_map.squeeze(-2).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).unsqueeze(-2)   # (N, H, W, 1, 3)
            if model.bkg_feats is not None:
                bkg_attn = attn[..., topk:, :]
                if config.models.normalize_topk_attn:
                    rgb = foreground_rgb * (1 - bkg_attn) + model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
                    bkg_mask = (model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn).squeeze()
                else:
                    rgb = foreground_rgb + model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
                    bkg_mask = (model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn).squeeze()
                rgb = rgb.squeeze(-2)
            else:
                rgb = foreground_rgb.squeeze(-2)
            foreground_rgb = foreground_rgb.squeeze()
        else:
            rgb = feature_map.squeeze(-2)

        rgb = model.last_act(rgb)
        rgb = torch.clamp(rgb, 0, 1)
    
    # Get depth
    # note: depth_np vs cur_depth coord_scale????????
    od = -rayo
    D = torch.sum(od * rayo)
    dists = torch.abs(torch.sum(selected_points.to(od.device) * od, -1) - D) / torch.norm(od)
    if model.bkg_feats is not None:
        dists = torch.cat([dists, torch.ones(N, H, W, model.bkg_feats.shape[0]).to(dists.device) * 0], dim=-1)
    cur_depth = (torch.sum(attn.squeeze(-1).to(od.device) * dists, dim=-1)) #.detach().cpu().squeeze().numpy().astype(np.float32)

    return {'rgb': rgb, 'depth': cur_depth}


def batchify_rays(rays_flat, model, config, chunk=1024*32, use_viewdirs=False, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    #print('BATCHIFY', chunk)
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        #print('ARGS CHUNK', chunk)
        ret = render_papr(rays_flat[i:i+chunk], model, use_viewdirs, config, None, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, intrinsic, model, config, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1., with_5_9=False, use_viewdirs=False, c2w_staticcam=None, 
                  rays_depth=None, **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      with_5_9: render with aspect ratio 5.33:9 (one third of 16:9)
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    ### DELETE ###
    #print('4 - Render Rays')

    #print(rays[0].shape)
    #print(rays[1].shape)

    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, intrinsic, c2w)
        if with_5_9:
            W_before = W
            W = int(H / 9. * 16. / 3.)
            if W % 2 != 0:
                W = W - 1
            start = (W_before - W) // 2
            rays_o = rays_o[:, start:start + W, :]
            rays_d = rays_d[:, start:start + W, :]
    elif rays.shape[0] == 2:
        # use provided ray batch
        rays_o, rays_d = rays
    else:
        rays_o, rays_d, rays_depth = rays
    
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, intrinsic, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    #print('RAYS D', rays_d.shape)
    #print('RAYS O', rays_o.shape)

    sh = rays_d.shape # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    if rays_depth is not None:
        rays_depth = torch.reshape(rays_depth, [-1,3]).float()
        rays = torch.cat([rays, rays_depth], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, model, config, chunk, use_viewdirs, **kwargs)
    all_ret['rgb'] = torch.reshape(all_ret['rgb'], sh)
    all_ret['depth'] = torch.reshape(all_ret['depth'], sh[:-1])
    # for k in all_ret:
    #     # k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
    #     # all_ret[k] = torch.reshape(all_ret[k], k_sh)
    #     print(k)
    #     print(all_ret[k].shape)
    #     print(sh)
    #     if all_ret[k].dim() == 4:
    #         all_ret[k] = torch.reshape(all_ret[k], sh)
    #     else:
    #         all_ret[k] = torch.reshape(all_ret[k], sh[:-1])

    # k_extract = ['rgb', 'depth']
    # ret_list = [all_ret[k] for k in k_extract]
    # ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    # return ret_list + [ret_dict]
    return all_ret['rgb'], all_ret['depth']

### PAPR ###
def create_papr(args, config, scene_render_params):
    """Instantiate PAPR's model based on the provided arguments."""
    ### DELETE ###
    #print('3 - Create PAPR')

    # Initialize the PAPR model with specified architecture parameters
    model = get_model(config, device)
    model = model.to(device)

    grad_vars = list(model.parameters())
    grad_names = [name for name, _ in model.named_parameters()]

    # Create the optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0  # For tracking training iterations; adjust based on checkpoint loading if implemented

    # Load checkpoint if applicable (adjust according to PAPR's checkpointing setup)
    # Placeholder for checkpoint loading logic
    # Load checkpoints
    ckpt = load_checkpoint(args)
    if ckpt is not None:
        start = ckpt['global_step']
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_my_state_dict(ckpt['network_fn_state_dict'])

    # Prepare rendering parameters, adjusting for PAPR's specifics
    render_kwargs_train = {
        #'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'raw_noise_std': args.raw_noise_std,
    }
    render_kwargs_train.update(scene_render_params)

    # Setup for test-time rendering; may include disabling perturbations and noise
    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return model, render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, grad_names



# Define the training function
def train_papr(images, depths, valid_depths, poses, intrinsics, i_split, args, config, scene_sample_params, lpips_alex, gt_depths, gt_valid_depths, all_depth_hypothesis, is_init_scales=False, scales_init=None, shifts_init=None):
    """
    Train a Proximity Attention Point Rendering (PAPR) model.
    
    Parameters:
    - images: Array of input images.
    - depths, valid_depths: Arrays of depth maps and their validity masks.
    - poses, intrinsics: Camera poses and intrinsics for each image.
    - i_split: Indices splitting the dataset into training, validation, and test sets.
    - args: Configuration arguments for the model and training process.
    - scene_sample_params: Parameters for scene sampling, including near and far plane distances.
    - lpips_alex: Pretrained LPIPS model for perceptual loss calculation.
    - gt_depths, gt_valid_depths: Ground truth depth maps and their validity masks, if available.
    - all_depth_hypothesis: Depth hypotheses for all images.
    - is_init_scales, scales_init, shifts_init: Initial scale and shift parameters for depth hypothesis adjustment.
    """
    #print('2 - Train PAPR')
    
    # Set seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    # Initialize TensorBoard for logging
    tb = SummaryWriter(log_dir=os.path.join("runs", args.expname))
    
    # Extract scene sampling parameters
    near, far = scene_sample_params['near'], scene_sample_params['far']
    
    # Image dimensions
    H, W = images.shape[1:3]
    
    # Dataset indices for train, validation, test, and video
    i_train, i_val, i_test, i_video = i_split
    print('TRAIN views are', i_train)
    print('VAL views are', i_val)
    print('TEST views are', i_test)

    # Use ground truth depth for validation and test sets, if available
    if gt_depths is not None:
        depths[i_test] = gt_depths[i_test]
        valid_depths[i_test] = gt_valid_depths[i_test]
        depths[i_val] = gt_depths[i_val]
        valid_depths[i_val] = gt_valid_depths[i_val]

    # Prepare data for training
    i_relevant_for_training = np.concatenate((i_train, i_val), 0)
    
    # Ensure there is a test set
    if len(i_test) == 0:
        print("Error: There is no test set")
        exit()
    
    # Use test set for validation if no validation set is present
    if len(i_val) == 0:
        print("Warning: There is no validation set, test set is used instead")
        i_val = i_test
        i_relevant_for_training = np.concatenate((i_relevant_for_training, i_val), 0)

    # Keep test data on CPU until needed
    test_images = images[i_test]
    test_depths = depths[i_test]
    test_valid_depths = valid_depths[i_test]
    test_poses = poses[i_test]
    test_intrinsics = intrinsics[i_test]
    i_test = i_test - i_test[0]

    # Move training data to GPU
    images = torch.Tensor(images[i_relevant_for_training]).to(device)
    depths = torch.Tensor(depths[i_relevant_for_training]).to(device)
    valid_depths = torch.Tensor(valid_depths[i_relevant_for_training]).bool().to(device)
    poses = torch.Tensor(poses[i_relevant_for_training]).to(device)
    intrinsics = torch.Tensor(intrinsics[i_relevant_for_training]).to(device)
    all_depth_hypothesis = torch.Tensor(all_depth_hypothesis).to(device)

    # Initialize the PAPR model and other training components
    # This function needs to be adapted or created for PAPR
    model, render_kwargs_train, render_kwargs_test, start, papr_grad_vars, optimizer, papr_grad_names = create_papr(args, config, scene_sample_params)
    
    ################################ TRAIN ################################
    ##### Initialize depth scale and shift
    DEPTH_SCALES = torch.autograd.Variable(torch.ones((images.shape[0], 1), dtype=torch.float, device=images.device)*args.scale_init, requires_grad=True)
    DEPTH_SHIFTS = torch.autograd.Variable(torch.ones((images.shape[0], 1), dtype=torch.float, device=images.device)*args.shift_init, requires_grad=True)

    print(DEPTH_SCALES)
    print()
    print(DEPTH_SHIFTS)
    print()
    print(DEPTH_SCALES.shape)
    print(DEPTH_SHIFTS.shape)

    optimizer_ss = torch.optim.Adam(params=(DEPTH_SCALES, DEPTH_SHIFTS,), lr=args.scaleshift_lr)
    
    print("Initialized scale and shift.")
    ################################

    # create camera embedding function
    embedcam_fn = None

    # optimize nerf
    print('Begin')
    N_iters = args.num_iterations + 1
    global_step = start
    start = start + 1

    init_learning_rate = args.lrate
    old_learning_rate = init_learning_rate

    # if args.cimle_white_balancing and args.load_pretrained:
    if args.load_pretrained:
        path = args.pretrained_dir
        ckpts = [os.path.join(path, f) for f in sorted(os.listdir(path)) if '000.tar' in f]
        print('Found ckpts', ckpts)
        ckpt_path = ckpts[-1]
        print('Reloading pretrained model from', ckpt_path)

        ckpt = torch.load(ckpt_path)

        model_dict = render_kwargs_train["network"].state_dict()
        keys = {k: v for k, v in ckpt['network_state_dict'].items() if k in model_dict} 

        print(len(keys.keys()))

        print("Num keys loaded:")
        model_dict.update(keys)

        ## Load scale and shift
        DEPTH_SHIFTS = torch.load(ckpt_path)["depth_shifts"]
        DEPTH_SCALES = torch.load(ckpt_path)["depth_scales"] 

        print("Scales:")
        print(DEPTH_SCALES)
        print()
        print("Shifts:")
        print(DEPTH_SHIFTS)

        print("Loaded depth shift/scale from pretrained model.")
        ########################################
        ########################################
    
    ##############
    # TRAIN LOOP #
    ##############
    # N_iters = 2
    for i in trange(start, N_iters):

        ### Scale the hypotheses by scale and shift
        img_i = np.random.choice(i_train)

        curr_scale = DEPTH_SCALES[img_i]
        curr_shift = DEPTH_SHIFTS[img_i]

        ## Scale and shift
        batch_rays, target_s, target_d, target_vd, img_i, target_h, space_carving_mask, curr_cached_u = get_ray_batch_from_one_image_hypothesis_idx(H, W, img_i, images, depths, valid_depths, poses, intrinsics, all_depth_hypothesis, args, None, None)

        target_h = target_h*curr_scale + curr_shift 

        if args.input_ch_cam > 0:
            render_kwargs_train['embedded_cam'] = embedcam_fn[img_i]

        target_d = target_d.squeeze(-1)

        render_kwargs_train["cached_u"] = None

        rgb, depth = render(H, W, None, model, config, chunk=args.chunk, rays=batch_rays, retraw=True,  is_joint=args.is_joint, **render_kwargs_train)

        # compute loss and optimize
        optimizer.zero_grad()
        optimizer_ss.zero_grad()
        img_loss = img2mse(rgb, target_s)
        psnr = mse2psnr(img_loss)
        
        loss = img_loss

        if args.space_carving_weight>0.: #and i>args.warm_start_papr:
            space_carving_loss = compute_space_carving_loss_papr(depth, target_h, is_joint=args.is_joint, norm_p=args.norm_p, threshold=args.space_carving_threshold, mask=space_carving_mask)
            loss = loss + args.space_carving_weight * space_carving_loss
        else:
            space_carving_loss = torch.mean(torch.zeros([target_h.shape[0]]).to(target_h.device))

        loss.backward()

         ### Update learning rate
        learning_rate = get_learning_rate(init_learning_rate, i, args.decay_step, args.decay_rate, staircase=True)
        if old_learning_rate != learning_rate:
            update_learning_rate(optimizer, learning_rate)
            old_learning_rate = learning_rate

        optimizer.step()

        ### Don't optimize scale shift for the last 100k epochs, check whether the appearance will crisp
        if i < args.freeze_ss:
            optimizer_ss.step()

        ### Update camera embeddings
        if args.input_ch_cam > 0 and args.opt_ch_cam:
            optimizer_latent.step() 

        # write logs
        if i%args.i_weights==0:
            path = os.path.join(args.ckpt_dir, args.expname, '{:06d}.tar'.format(i))
            save_dict = {
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),}
            
            if args.input_ch_cam > 0:
                save_dict['embedded_cam'] = embedcam_fn

            save_dict['depth_shifts'] = DEPTH_SHIFTS
            save_dict['depth_scales'] = DEPTH_SCALES

            torch.save(save_dict, path)
            print('Saved checkpoints at', path)

        if i%args.i_print==0:
            tb.add_scalars('mse', {'train': img_loss.item()}, i)

            if args.space_carving_weight > 0.:
                tb.add_scalars('space_carving_loss', {'train': space_carving_loss.item()}, i)

            tb.add_scalars('psnr', {'train': psnr.item()}, i)

            scale_mean = torch.mean(DEPTH_SCALES[i_train])
            shift_mean = torch.mean(DEPTH_SHIFTS[i_train])
            tb.add_scalars('depth_scale_mean', {'train': scale_mean.item()}, i)
            tb.add_scalars('depth_shift_mean', {'train': shift_mean.item()}, i) 

            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}  MSE: {img_loss.item()} Space carving: {space_carving_loss.item()}")
        
        if i%args.i_img==0:
            # visualize 2 train images
            _, images_train = render_images_with_metrics(model, 2, i_train, images, depths, valid_depths, \
                poses, H, W, intrinsics, lpips_alex, args, config, render_kwargs_test, embedcam_fn=embedcam_fn)
            tb.add_image('train_image',  torch.cat((
                torchvision.utils.make_grid(images_train["rgbs"], nrow=1), \
                torchvision.utils.make_grid(images_train["target_rgbs"], nrow=1), \
                torchvision.utils.make_grid(images_train["depths"], nrow=1), \
                torchvision.utils.make_grid(images_train["target_depths"], nrow=1)), 2), i)
            # compute validation metrics and visualize 8 validation images
            mean_metrics_val, images_val = render_images_with_metrics(model, 8, i_val, images, depths, valid_depths, \
                poses, H, W, intrinsics, lpips_alex, args, config, render_kwargs_test)
            tb.add_scalars('mse', {'val': mean_metrics_val.get("img_loss")}, i)
            tb.add_scalars('psnr', {'val': mean_metrics_val.get("psnr")}, i)
            tb.add_scalar('ssim', mean_metrics_val.get("ssim"), i)
            tb.add_scalar('lpips', mean_metrics_val.get("lpips"), i)
            if mean_metrics_val.has("depth_rmse"):
                tb.add_scalar('depth_rmse', mean_metrics_val.get("depth_rmse"), i)
            tb.add_image('val_image',  torch.cat((
                torchvision.utils.make_grid(images_val["rgbs"], nrow=1), \
                torchvision.utils.make_grid(images_val["target_rgbs"], nrow=1), \
                torchvision.utils.make_grid(images_val["depths"], nrow=1), \
                torchvision.utils.make_grid(images_val["target_depths"], nrow=1)), 2), i)

        # test at the last iteration
        if (i + 1) == N_iters:
            torch.cuda.empty_cache()
            images = torch.Tensor(test_images).to(device)
            depths = torch.Tensor(test_depths).to(device)
            valid_depths = torch.Tensor(test_valid_depths).bool().to(device)
            poses = torch.Tensor(test_poses).to(device)
            intrinsics = torch.Tensor(test_intrinsics).to(device)
            mean_metrics_test, images_test = render_images_with_metrics(model, None, i_test, images, depths, valid_depths, \
                poses, H, W, intrinsics, lpips_alex, args, config, render_kwargs_test)
            write_images_with_metrics(images_test, mean_metrics_test, far, args)
            tb.flush()

        global_step += 1


def run_papr():
    ### DELETE ###
    print('1 - run papr')

    # Parse configuration and command-line arguments
    parser = config_parser()
    args = parser.parse_args()

    with open(args.opt, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = dict_to_namespace(config_dict)

    # Set up experiment name and save configuration arguments to a JSON file for reproducibility
    if args.task == "train":
        if args.expname is None:
            args.expname = "{}_{}".format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), args.scene_id)
        args_file = os.path.join(args.ckpt_dir, args.expname, 'args.json')
        os.makedirs(os.path.join(args.ckpt_dir, args.expname), exist_ok=True)
        with open(args_file, 'w') as af:
            json.dump(vars(args), af, indent=4)

    # Print configuration arguments for verification
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    # Determine the number of GPUs available and print it out
    args.n_gpus = torch.cuda.device_count()
    print(f"Using {args.n_gpus} GPU(s).")

    # Load dataset including images, depths, camera poses, etc., from a specified directory
    scene_data_dir = os.path.join(args.data_dir, args.scene_id)
    images, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, gt_depths, gt_valid_depths, all_depth_hypothesis = load_scene_scannet(scene_data_dir, args.cimle_dir, args.num_hypothesis, 'transforms_train.json')

    i_train, i_val, i_test, i_video = i_split

    # Compute boundaries of 3D space
    max_xyz = torch.full((3,), -1e6)
    min_xyz = torch.full((3,), 1e6)
    for idx_train in i_train:
        rays_o, rays_d = get_rays(H, W, torch.Tensor(intrinsics[idx_train]), torch.Tensor(poses[idx_train])) # (H, W, 3), (H, W, 3)
        points_3D = rays_o + rays_d * far # [H, W, 3]
        max_xyz = torch.max(points_3D.view(-1, 3).amax(0), max_xyz)
        min_xyz = torch.min(points_3D.view(-1, 3).amin(0), min_xyz)
    args.bb_center = (max_xyz + min_xyz) / 2.
    args.bb_scale = 2. / (max_xyz - min_xyz).max()
    print("Computed scene boundaries: min {}, max {}".format(min_xyz, max_xyz))

    # Additional scene parameters, might include precomputed samples for importance sampling
    scene_sample_params = {
        'precomputed_z_samples': None,
        'near': near,
        'far': far,
    }

    # Initialize the LPIPS model for perceptual loss calculation, if used
    lpips_alex = LPIPS()

    # Execute training procedure
    if args.task == "train":
        train_papr(images, depths, valid_depths, poses, intrinsics, i_split, args, config, scene_sample_params, lpips_alex, gt_depths, gt_valid_depths, all_depth_hypothesis)
    exit()

    # For testing and video rendering, create a PAPR model with parameters set for inference (requires_grad = False)
    _, render_kwargs_test, _, papr_grad_vars, _, papr_grad_names = create_papr(args, scene_sample_params)

    for param in nerf_grad_vars:
        param.requires_grad = False

    # render test set and compute statistics
    if "test" in args.task: 
        with_test_time_optimization = False
        if args.task == "test_opt":
            with_test_time_optimization = True
        images = torch.Tensor(images[i_test]).to(device)
        if gt_depths is None:
            depths = torch.Tensor(depths[i_test]).to(device)
            valid_depths = torch.Tensor(valid_depths[i_test]).bool().to(device)
        else:
            depths = torch.Tensor(gt_depths[i_test]).to(device)
            valid_depths = torch.Tensor(gt_valid_depths[i_test]).bool().to(device)
        poses = torch.Tensor(poses[i_test]).to(device)
        intrinsics = torch.Tensor(intrinsics[i_test]).to(device)
        i_test = i_test - i_test[0]
        mean_metrics_test, images_test = render_images_with_metrics(None, i_test, images, depths, valid_depths, poses, H, W, intrinsics, lpips_alex, args, \
            render_kwargs_test, with_test_time_optimization=with_test_time_optimization)
        write_images_with_metrics(images_test, mean_metrics_test, far, args, with_test_time_optimization=with_test_time_optimization)
    elif args.task == "video":
        vposes = torch.Tensor(poses[i_video]).to(device)
        vintrinsics = torch.Tensor(intrinsics[i_video]).to(device)
        render_video(vposes, H, W, vintrinsics, str(0), args, render_kwargs_test)


if __name__=='__main__':

    ### DELETE ###
    print('Started')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    run_papr()