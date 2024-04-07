import numpy as np
import json
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import yaml
from models import get_model 
from data.load_scene import load_scene_scannet
from papr_helpers import compute_space_carving_loss_papr, get_papr_embedder # Adjust as necessary
from papr_scade_utils import config_parser, img2mse, mse2psnr, update_learning_rate, get_learning_rate, get_rays
from prior_utils import get_ray_batch_from_one_image_hypothesis_idx
from torch.optim import Adam
from tqdm import tqdm, trange
from torchvision import utils as torchvision_utils
from lpips import LPIPS # Assuming you're using a perceptual loss; adjust as necessary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn

def render_papr(ray_batch, use_viewdirs, model, c2w=None):

    ### DELETE ###
    print('5 - Render PAPR')

    # Get information from the batch
    N_rays = ray_batch.shape[0]
    rayo, rayd = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each

    # get the feature map and attn
    # copied from PAPR test.py test_step line 49
    N, H, W, _ = rayd.shape
    num_pts, _ = model.points.shape

    rayo = rayo.to(device)
    rayd = rayd.to(device)
    c2w = c2w.to(device)

    topk = min([num_pts, model.select_k])
    selected_points = torch.zeros(1, H, W, topk, 3)

    bkg_seq_len_attn = 0
    tx_opt = args.models.transformer
    feat_dim = tx_opt.embed.d_ff_out if tx_opt.embed.share_embed else tx_opt.embed.value.d_ff_out
    if model.bkg_feats is not None:
        bkg_seq_len_attn = model.bkg_feats.shape[0]
    feature_map = torch.zeros(N, H, W, 1, feat_dim).to(device)
    attn = torch.zeros(N, H, W, topk + bkg_seq_len_attn, 1).to(device)
    
    with torch.no_grad():
        for height_start in range(0, H, args.test.max_height):
            for width_start in range(0, W, args.test.max_width):
                height_end = min(height_start + args.test.max_height, H)
                width_end = min(width_start + args.test.max_width, W)

                feature_map[:, height_start:height_end, width_start:width_end, :, :], \
                attn[:, height_start:height_end, width_start:width_end, :, :] = model.evaluate(rayo, rayd[:, height_start:height_end, width_start:width_end], c2w, step=resume_step)

                selected_points[:, height_start:height_end, width_start:width_end, :, :] = model.selected_points

        if args.models.use_renderer:
            foreground_rgb = model.renderer(feature_map.squeeze(-2).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).unsqueeze(-2)   # (N, H, W, 1, 3)
            if model.bkg_feats is not None:
                bkg_attn = attn[..., topk:, :]
                if args.models.normalize_topk_attn:
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
    cur_depth = (torch.sum(attn.squeeze(-1).to(od.device) * dists, dim=-1)).detach().cpu().squeeze().numpy().astype(np.float32)

    return rbg, cur_depth


def batchify_rays(rays_flat, chunk=1024*32, use_viewdirs=False, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_papr(rays_flat[i:i+chunk], use_viewdirs, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, intrinsic, chunk=1024*32, rays=None, c2w=None, ndc=True,
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
    print('4 - Render Rays')

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
    all_ret = batchify_rays(rays, chunk, use_viewdirs, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb', 'depth']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def create_papr(args, config, scene_render_params):
    """Instantiate PAPR's model based on the provided arguments."""
    ### DELETE ###
    print('3 - Create PAPR')
    # Embedding functions for positional encoding of inputs and view directions
    embed_fn, input_ch = get_papr_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_papr_embedder(args.multires_views, args.i_embed)
    
    # Specify output channels; adjust based on PAPR's requirements
    output_ch = 5  # This may need to be adjusted for PAPR

    # Initialize the PAPR model with specified architecture parameters
    model = get_model(config, device)

    # Utilize DataParallel for multi-GPU support
    model = nn.DataParallel(model).to(args.device)
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
        'network_query_fn': network_query_fn,
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

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, grad_names



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
    print('2 - Train PAPR')

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
    render_kwargs_train, render_kwargs_test, start, papr_grad_vars, optimizer, papr_grad_names = create_papr(args, config, scene_sample_params)
    
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

        rgb, depth = render(H, W, None, chunk=args.chunk, rays=batch_rays, verbose=i < 10, retraw=True,  is_joint=args.is_joint, **render_kwargs_train)

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
            _, images_train = render_images_with_metrics(2, i_train, images, depths, valid_depths, \
                poses, H, W, intrinsics, lpips_alex, args, render_kwargs_test, embedcam_fn=embedcam_fn)
            tb.add_image('train_image',  torch.cat((
                torchvision.utils.make_grid(images_train["rgbs"], nrow=1), \
                torchvision.utils.make_grid(images_train["target_rgbs"], nrow=1), \
                torchvision.utils.make_grid(images_train["depths"], nrow=1), \
                torchvision.utils.make_grid(images_train["target_depths"], nrow=1)), 2), i)
            # compute validation metrics and visualize 8 validation images
            mean_metrics_val, images_val = render_images_with_metrics(8, i_val, images, depths, valid_depths, \
                poses, H, W, intrinsics, lpips_alex, args, render_kwargs_test)
            tb.add_scalars('mse', {'val': mean_metrics_val.get("img_loss")}, i)
            tb.add_scalars('psnr', {'val': mean_metrics_val.get("psnr")}, i)
            tb.add_scalar('ssim', mean_metrics_val.get("ssim"), i)
            tb.add_scalar('lpips', mean_metrics_val.get("lpips"), i)
            if mean_metrics_val.has("depth_rmse"):
                tb.add_scalar('depth_rmse', mean_metrics_val.get("depth_rmse"), i)
            if 'rgbs0' in images_val:
                tb.add_scalars('mse0', {'val': mean_metrics_val.get("img_loss0")}, i)
                tb.add_scalars('psnr0', {'val': mean_metrics_val.get("psnr0")}, i)
            if 'rgbs0' in images_val:
                tb.add_image('val_image',  torch.cat((
                    torchvision.utils.make_grid(images_val["rgbs"], nrow=1), \
                    torchvision.utils.make_grid(images_val["rgbs0"], nrow=1), \
                    torchvision.utils.make_grid(images_val["target_rgbs"], nrow=1), \
                    torchvision.utils.make_grid(images_val["depths"], nrow=1), \
                    torchvision.utils.make_grid(images_val["depths0"], nrow=1), \
                    torchvision.utils.make_grid(images_val["target_depths"], nrow=1)), 2), i)
            else:
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
            mean_metrics_test, images_test = render_images_with_metrics(None, i_test, images, depths, valid_depths, \
                poses, H, W, intrinsics, lpips_alex, args, render_kwargs_test)
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
        config = yaml.safe_load(f)

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