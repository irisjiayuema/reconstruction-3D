import torch
import numpy as np
from papr_scade_utils import get_rays

################################
# From SCADE run_scade_scannet #
################################

def select_coordinates(coords, N_rand):
    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
    select_coords = coords[select_inds].long()  # (N_rand, 2)
    return select_coords

# Gets depth from prior
def get_ray_batch_from_one_image_hypothesis_idx(H, W, img_i, images, depths, valid_depths, poses, intrinsics, all_hypothesis, args, space_carving_idx=None, cached_u=None):
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W), indexing='ij'), -1)  # (H, W, 2)
    # img_i = np.random.choice(i_train)
    
    target = images[img_i]
    target_depth = depths[img_i]
    target_valid_depth = valid_depths[img_i]
    pose = poses[img_i]
    intrinsic = intrinsics[img_i, :]

    target_hypothesis = all_hypothesis[img_i]

    rays_o, rays_d = get_rays(H, W, intrinsic, pose)  # (H, W, 3), (H, W, 3)
    select_coords = select_coordinates(coords, args.N_rand)

    # print(rays_d.shape)
    # print(rays_o.shape)

    # rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    # rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    # target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    # target_d = target_depth[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1) or (N_rand, 2)
    # target_vd = target_valid_depth[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1)
    # target_h = target_hypothesis[:, select_coords[:, 0], select_coords[:, 1]]

    target_s = target
    target_d = target_depth
    target_vd = target_valid_depth
    target_h = target_hypothesis
    
    # print(H, W)
    # print(args.N_rand)
    # print(rays_d[0].shape)
    # print(rays_o.shape)

    if space_carving_idx is not None:
        # print(space_carving_idx.shape)
        # print(space_carving_idx[img_i, select_coords[:, 0], select_coords[:, 1]].shape)
        target_hypothesis  = target_hypothesis.repeat(1, 1, 1, space_carving_idx.shape[-1])

        curr_space_carving_idx = space_carving_idx[img_i, select_coords[:, 0], select_coords[:, 1]]

        target_h_rays = target_hypothesis[ :, select_coords[:, 0], select_coords[:, 1]]

        target_h = torch.gather(target_h_rays, 1, curr_space_carving_idx.unsqueeze(0).long())


    if cached_u is not None:
        curr_cached_u = cached_u[img_i, select_coords[:, 0], select_coords[:, 1]]
    else:
        curr_cached_u = None

    if args.mask_corners:
        ### Initialize a masked image
        space_carving_mask = torch.ones((target.shape[0], target.shape[1]), dtype=torch.float, device=images.device)

        ### Mask out the corners
        num_pix_to_mask = 20
        space_carving_mask[:num_pix_to_mask, :num_pix_to_mask] = 0
        space_carving_mask[:num_pix_to_mask, -num_pix_to_mask:] = 0
        space_carving_mask[-num_pix_to_mask:, :num_pix_to_mask] = 0
        space_carving_mask[-num_pix_to_mask:, -num_pix_to_mask:] = 0

        space_carving_mask = space_carving_mask[select_coords[:, 0], select_coords[:, 1]]
    else:
        space_carving_mask = None

    batch_rays = torch.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)
    
    return batch_rays, target_s, target_d, target_vd, img_i, target_h, space_carving_mask, curr_cached_u
