'''
Mikaela Uy
mikacuy@stanford.edu
'''
import math
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "../"))
from data.finetune_dataset import FinetuneDataset, FinetuneDataset_wild
from lib.models.multi_depth_model_auxiv2 import *
from lib.configs.config import cfg, merge_cfg_from_file, print_configs
from lib.utils.net_tools import save_ckpt, load_ckpt
from tools.parse_arg_base import print_options
from lib.configs.config import cfg, merge_cfg_from_file, print_configs

## for dataloaders
import torch.utils.data

import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import torchvision.transforms as transforms

import argparse
from PIL import Image
import random
import imageio
import json


parser = argparse.ArgumentParser()

parser.add_argument("--logdir", default="../ambiguity_aware_prior_pretrained_model/", help="path to the log directory", type=str)
parser.add_argument("--ckpt", default="model.pth", help="checkpoint", type=str)

### For the dataset
parser.add_argument('--phase', type=str, default='test', help='Training flag')


parser.add_argument('--dataroot', default='../datasets/in_the_wild/lounge/train/', help='Root dir for dataset')
parser.add_argument('--dump_dir', default= "dump_prior_samples/", type=str)

parser.add_argument('--backbone', default= "resnext101", type=str)
parser.add_argument('--d_latent', default= 32, type=int)
parser.add_argument('--num_samples', default= 20, type=int)
parser.add_argument('--rescaled', default=False, type=bool)

parser.add_argument('--ada_version', default= "v2", type=str)
parser.add_argument('--cimle_version', default= "enc", type=str)
parser.add_argument('--import_from_logdir', default=False, type=bool)
parser.add_argument('--visu_all', default=True, type=bool)
parser.add_argument('--seed_num', default= 0, type=int)

parser.add_argument('--default_scale', default= 0.5, type=float)
parser.add_argument('--default_shift', default= 0.0, type=float)

parser.add_argument('--is_nsvf', default=False, type=bool)
parser.add_argument('--is_wild', default=True, type=bool)


FLAGS = parser.parse_args()
LOG_DIR = FLAGS.logdir
CKPT = FLAGS.ckpt

IMPORT_FROM_LOGDIR = FLAGS.import_from_logdir
VISU_ALL = FLAGS.visu_all
SEED_NUM = FLAGS.seed_num
torch.manual_seed(SEED_NUM)
np.random.seed(SEED_NUM)
random.seed(SEED_NUM)

IS_NSVF = FLAGS.is_nsvf
IS_WILD = FLAGS.is_wild

#### Import from LOG_DIR files or from global files
if IMPORT_FROM_LOGDIR:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, "../", LOG_DIR))
    from data.load_dataset_distributed import MultipleDataLoaderDistributed, MultipleDatasetDistributed
    from lib.models.multi_depth_model_auxiv2 import *
    from lib.configs.config import cfg, merge_cfg_from_file, print_configs
    from lib.utils.net_tools import save_ckpt, load_ckpt
    from tools.parse_arg_base import print_options
    from lib.configs.config import cfg, merge_cfg_from_file, print_configs
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, "../"))
    from data.load_dataset_distributed import MultipleDataLoaderDistributed, MultipleDatasetDistributed
    from lib.models.multi_depth_model_auxiv2 import *
    from lib.configs.config import cfg, merge_cfg_from_file, print_configs
    from lib.utils.net_tools import save_ckpt, load_ckpt
    from tools.parse_arg_base import print_options
    from lib.configs.config import cfg, merge_cfg_from_file, print_configs

ADA_VERSION = FLAGS.ada_version
CIMLE_VERSION = FLAGS.cimle_version
print(CIMLE_VERSION)
print(ADA_VERSION)
print("===================")

DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
temp_fol = os.path.join(DUMP_DIR, "tmp")
if not os.path.exists(temp_fol): os.mkdir(temp_fol)
pc_fol = os.path.join(DUMP_DIR, "pc")
if not os.path.exists(pc_fol): os.mkdir(pc_fol)
gt_fol = os.path.join(DUMP_DIR, "gt")
if not os.path.exists(gt_fol): os.mkdir(gt_fol)

LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

RESCALED = FLAGS.rescaled
D_LATENT = FLAGS.d_latent
NUM_SAMPLE = FLAGS.num_samples

print("Rescaled: "+str(RESCALED))

### Merge config with current configs
merge_cfg_from_file(FLAGS)

if CIMLE_VERSION == "enc":
    model = RelDepthModel_cIMLE(d_latent=D_LATENT, version=ADA_VERSION)
else:
    model = RelDepthModel_cIMLE_decoder(d_latent=D_LATENT, version=ADA_VERSION)

model.cuda()

### Load model
model_dict = model.state_dict()

# CKPT_FILE = os.path.join(LOG_DIR, CKPT)
CKPT_FILE = os.path.join(LOG_DIR, CKPT)

if os.path.isfile(CKPT_FILE):
    print("loading checkpoint %s" % CKPT_FILE)
    checkpoint = torch.load(CKPT_FILE)

    checkpoint['model_state_dict'] = strip_prefix_if_present(checkpoint['model_state_dict'], "module.")
    depth_keys = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict} ## <--- some missing keys in the loaded model from the given model
    print(len(depth_keys))

    # Overwrite entries in the existing state dict
    model_dict.update(depth_keys)        

    # Load the new state dict
    model.load_state_dict(model_dict)


    print("Model loaded.")

else:
	print("Error: Model does not exist.")
	exit()

mean0, var0, mean1, var1, mean2, var2, mean3, var3 = load_mean_var_adain(os.path.join(LOG_DIR, "mean_var_adain.npy"), torch.device("cuda"))
model.set_mean_var_shifts(mean0, var0, mean1, var1, mean2, var2, mean3, var3)
print("Initialized adain mean and var.")

### Quantitative Metrics ####
def evaluate_rel_err(pred, gt, mask_invalid=None, scale=10.0 ):
    if type(pred).__module__ != np.__name__:
        pred = pred.cpu().numpy()
    if type(gt).__module__ != np.__name__:
        gt = gt.cpu().numpy()

    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    if pred.shape != gt.shape:
        logger.info('The shapes of dt and gt are not same!')
        return -1
    if mask_invalid is not None:
        gt = gt[~mask_invalid]
        pred = pred[~mask_invalid]


    mask = (gt > 1e-8)
    gt = gt[mask]
    pred = pred[mask]
    n_pxl = gt.size
    gt_scale = gt * scale
    pred_scale = pred * scale

    err_absRel = -1.
    err_squaRel = -1.
    err_silog = -1.
    err_delta1 = -1.
    err_whdr = -1.


    if gt_scale.size < 10:
        print('Valid pixel size:', gt_scale.size, 'Invalid evaluation!!!!')
        exit()
        return err_absRel, err_squaRel, err_silog, err_delta1, err_whdr

    #Mean Absolute Relative Error
    rel = np.abs(gt - pred) / gt# compute errors
    abs_rel_sum = np.sum(rel)
    err_absRel = np.float64(abs_rel_sum) / float(n_pxl)

    #Square Mean Relative Error
    s_rel = ((gt_scale - pred_scale) * (gt_scale - pred_scale)) / (gt_scale * gt_scale)# compute errors
    squa_rel_sum = np.sum(s_rel)
    err_squaRel = np.float64(squa_rel_sum) / float(n_pxl)

    # Scale invariant error
    diff_log = np.log(pred_scale) - np.log(gt_scale)
    diff_log_sum = np.sum(diff_log)
    err_silog = np.float64(diff_log_sum)/ float(n_pxl)

    #Delta
    gt_pred = gt_scale / pred_scale
    pred_gt = pred_scale / gt_scale
    gt_pred = np.reshape(gt_pred, (1, -1))
    pred_gt = np.reshape(pred_gt, (1, -1))
    gt_pred_gt = np.concatenate((gt_pred, pred_gt), axis=0)
    ratio_max = np.amax(gt_pred_gt, axis=0)

    delta_1_sum = np.sum(ratio_max < 1.25)
    err_delta1 = np.float64(delta_1_sum)/ float(n_pxl)

    # WHDR error
    whdr_err_sum, eval_num = weighted_human_disagreement_rate(gt_scale, pred_scale)
    err_whdr = np.float64(whdr_err_sum)/ float(eval_num)

    return err_absRel, err_squaRel, err_silog, err_delta1, err_whdr


def weighted_human_disagreement_rate(gt, pred):
    p12_index = select_index(gt)
    gt_reshape = np.reshape(gt, gt.size)
    pred_reshape = np.reshape(pred, pred.size)
    mask = gt > 0
    gt_p1 = gt_reshape[mask][p12_index['p1']]
    gt_p2 = gt_reshape[mask][p12_index['p2']]
    pred_p1 = pred_reshape[mask][p12_index['p1']]
    pred_p2 = pred_reshape[mask][p12_index['p2']]

    p12_rank_gt = np.zeros_like(gt_p1)
    p12_rank_gt[gt_p1 > gt_p2] = 1
    p12_rank_gt[gt_p1 < gt_p2] = -1

    p12_rank_pred = np.zeros_like(gt_p1)
    p12_rank_pred[pred_p1 > pred_p2] = 1
    p12_rank_pred[pred_p1 < pred_p2] = -1

    err = np.sum(p12_rank_gt != p12_rank_pred)
    valid_pixels = gt_p1.size
    return err, valid_pixels


def select_index(gt_depth, select_size=10000):
    valid_size = np.sum(gt_depth>0)
    try:
        p = np.random.choice(valid_size, select_size*2, replace=False)
    except:
        p = np.random.choice(valid_size, select_size*2*2, replace=True)
    np.random.shuffle(p)
    p1 = p[0:select_size*2:2]
    p2 = p[1:select_size*2:2]

    p12_index = {'p1': p1, 'p2': p2}
    return p12_index


### Get valid gt depths 
def transform_shift_scale(depth, valid_threshold=-1e-8, max_threshold=1e8):
    mask = (depth > valid_threshold) & (depth < max_threshold)     
    gt_maskbatch = depth[mask]

    # Get mean and standard deviation
    data_mean = []
    data_std_dev = []
    for i in range(depth.shape[0]):
        gt_i = depth[i]
        mask = gt_i > 0
        depth_valid = gt_i[mask]
        depth_valid = depth_valid[:5]
        if depth_valid.shape[0] < 10:
            data_mean.append(torch.tensor(0).cuda())
            data_std_dev.append(torch.tensor(1).cuda())
            continue
        size = depth_valid.shape[0]
        depth_valid_sort, _ = torch.sort(depth_valid, 0)
        depth_valid_mask = depth_valid_sort[int(size*0.1): -int(size*0.1)]
        data_mean.append(depth_valid_mask.mean())
        data_std_dev.append(depth_valid_mask.std())
    data_mean = torch.stack(data_mean, dim=0).cuda()
    data_std_dev = torch.stack(data_std_dev, dim=0).cuda()

    gt_trans = (depth - data_mean[:, None, None, None]) / (data_std_dev[:, None, None, None] + 1e-8)
    
    return gt_trans

### Tranform pred to fit the ground truth

#### Fit with regularization ####
def recover_metric_depth(pred, gt):
    if type(pred).__module__ == torch.__name__:
        pred = pred.cpu().numpy()
    if type(gt).__module__ == torch.__name__:
        gt = gt.cpu().numpy()
    gt = gt.squeeze()
    pred = pred.squeeze()
    

    mask = (gt > 0.5)

    if np.sum(mask) == 0 :
        print("Warning: No SfM points found.")
        return pred, FLAGS.default_scale, FLAGS.default_shift

    gt_mask = gt[mask]
    pred_mask = pred[mask]
    a, b = np.polyfit(pred_mask, gt_mask, deg=1)
    
    pred_metric = a * pred + b

    # pred_metric[~mask] = 0.

    return pred_metric, a, b

#### Image transform #####
def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img

def remap_color_to_depth(depth_img):
    gray_values = np.arange(256, dtype=np.uint8)
    color_values = map(tuple, cv2.applyColorMap(gray_values, cv2.COLORMAP_TURBO).reshape(256, 3))
    color_to_gray_map = dict(zip(color_values, gray_values))

    depth = np.apply_along_axis(lambda bgr: color_to_gray_map[tuple(bgr)], 2, depth_img)
    return depth    
##############################

### Compute RSME ###
def compute_rmse(prediction, target):
    return np.sqrt(np.mean(np.square(prediction - target)))

def per_pixel_error(prediction, target):
    return np.sqrt(np.square(prediction - target))

### Dataset

### Dataloader
datapath = FLAGS.dataroot

if not IS_WILD:
    dataset = FinetuneDataset(datapath, "processed", is_nsvf=IS_NSVF, split="test", data_aug=False)
else:
    dataset = FinetuneDataset_wild(datapath, "processed", is_nsvf=IS_NSVF, split="test", data_aug=False)


### Create output dir for the multiple hypothesis
hypothesis_outdir = os.path.join(FLAGS.dataroot, "leres_cimle", DUMP_DIR)
if not os.path.exists(hypothesis_outdir): os.makedirs(hypothesis_outdir)

##### Also load intrinsics and depth scale for the dataset. #####
json_fname =  os.path.join(datapath, '../transforms_train.json')
with open(json_fname, 'r') as fp:
    meta = json.load(fp)

depth_scaling_factor = float(meta['depth_scaling_factor'])
#################################################################

#### Evaluation ######
zcache_dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=1,
    num_workers=0,
    shuffle=False)
print(len(zcache_dataloader))

mini_batch_size = 5
num_sets = int(NUM_SAMPLE/mini_batch_size)
true_num_samples = num_sets*mini_batch_size # just take the floor

### For quantitative evaluation
num_evaluated = 0

model.eval()

### Focal length for scannet
if not IS_NSVF:
    f = 577.870605
else:
    ### Check this for other models
    f = 1111.111

with torch.no_grad():

    all_scales = []
    all_shifts = []

    for i, data in enumerate(zcache_dataloader):

        batch_size = data['rgb'].shape[0]
        C = data['rgb'].shape[1]
        H = data['rgb'].shape[2]
        W = data['rgb'].shape[3]

        ### Repeat for the number of samples
        num_images = data['rgb'].shape[0]
        data['rgb'] = data['rgb'].unsqueeze(1).repeat(1,mini_batch_size, 1, 1, 1)
        data['rgb'] = data['rgb'].view(-1, C, H, W)


        rgb = torch.clone(data['rgb'][0]).permute(1, 2, 0).to("cpu").detach().numpy() 
        rgb = rgb[:, :, ::-1] ## dataloader is bgr
        rgb = 255 * (rgb - rgb.min()) / (rgb.max() - rgb.min())
        rgb = np.array(rgb, np.int)


        ### Raw gt depth ###
        curr_depth_path = data['B_paths'][0]

        depth_img = cv2.imread(curr_depth_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
        valid_depth = depth_img > 0.5

        if not IS_NSVF:
            ## Scannet depth
            depth_img = (depth_img/depth_scaling_factor).astype(np.float32)
        else:
            depth_img = remap_color_to_depth(depth_img)
            depth_img = depth_img.astype(float)

        orig_shape = depth_img.shape

        ### For computing scale and shift
        depth_orig_size = depth_img.copy()
        depth_img = cv2.resize(depth_img, (448, 448), interpolation=cv2.INTER_NEAREST)

        #### Get image focal length
        frame = meta['frames'][i]
        fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
        intrinsics = np.array((fx, fy, cx, cy))
        ###########################        

        ### Load sparse SfM depth
        curr_sfm_depth_path = data['C_paths'][0]
        sfm_depth_img = cv2.imread(curr_sfm_depth_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
        sfm_depth_img = (sfm_depth_img/depth_scaling_factor).astype(np.float32)

        ### Filter out SfM points that are beyond the far plane
        sfm_depth_img[sfm_depth_img > meta["far"]] = 0.0
        valid_sfm_depth = sfm_depth_img > 0.5


        ### Iterate over the minibatch
        image_fname = []

        if i%10==0  or VISU_ALL:      
            img_name = "image" + str(i)
            cv2.imwrite(os.path.join(temp_fol, img_name+"-raw.png"), rgb)
            image_fname.append(os.path.join(temp_fol, img_name+"-raw.png"))
            img_name = "image" + str(i) + "_gt"
            reconstruct_depth_intrinsics(depth_img, rgb, gt_fol, img_name, intrinsics)


        sfm_image_scales = []
        sfm_image_shifts = []

        all_pred_depths = []

        for k in range(num_sets):

            ## Hard coded d_latent
            z = torch.normal(0.0, 1.0, size=(num_images, mini_batch_size, D_LATENT))
            z = z.view(-1, D_LATENT).cuda()

            pred_depth = model.inference(data, z, rescaled=RESCALED)

            for s in range(mini_batch_size):
                curr_pred_depth = pred_depth[s]
                curr_pred_depth = curr_pred_depth.to("cpu").detach().numpy().squeeze() 

                pred_depth_ori = curr_pred_depth
                # pred_depth_ori = cv2.resize(curr_pred_depth, (H, W))

                img_name = "image" + str(i) + "_" + str(k) + "_" + str(s)
                

                ### Resize first then compute for error
                curr_pred_depth_raw = cv2.resize(curr_pred_depth, (depth_orig_size.shape[1], depth_orig_size.shape[0])) ### check this resize function 

                ## SfM depth
                curr_pred_sfm_depth_metric, curr_sfm_scale, curr_sfm_shift = recover_metric_depth(curr_pred_depth_raw, sfm_depth_img)

                sfm_image_scales.append(curr_sfm_scale)
                sfm_image_shifts.append(curr_sfm_shift)

                if i%10==0  or VISU_ALL:
                    # save depth
                    plt.imsave(os.path.join(temp_fol, img_name+'-depth.png'), curr_pred_sfm_depth_metric, cmap='rainbow')
                    image_fname.append(os.path.join(temp_fol, img_name+'-depth.png'))

                    ### Output point cloud
                    reconstruct_depth_intrinsics(curr_pred_depth, rgb, pc_fol, img_name, intrinsics)

                    rgb_orig = cv2.imread(data['A_paths'][0])
                    reconstruct_depth_intrinsics(curr_pred_sfm_depth_metric, rgb_orig, pc_fol, img_name+"-sfmscaled", intrinsics)

                ### Change to using aligned depth instead
                all_pred_depths.append(curr_pred_sfm_depth_metric)

            #######

        all_pred_depths = np.stack(all_pred_depths)
        num_evaluated += 1       

        print("===============")
        print("i")
        ### Output to file
        for k in range(num_sets):
            for s in range(mini_batch_size):

                curr_depth = all_pred_depths[k*mini_batch_size+s]

                curr_rbg_name = data['A_paths'][0]

                if not IS_WILD:
                    fname = curr_rbg_name.split("/")[-1][:-4] + "_" + str(k*mini_batch_size+s) + ".npy"
                else:
                    fname = curr_rbg_name.split("/")[-1][:-5] + "_" + str(k*mini_batch_size+s) + ".npy"


                outfname = os.path.join(hypothesis_outdir, fname)

                np.save(outfname, np.array(curr_depth))                

                ## Check output --> debug
                depth = np.load(outfname).astype(np.float64)
                print(depth)
                print(depth.shape)
                print(outfname)
                print()


        if i%10==0  or VISU_ALL:
            ### Collate and output to a single image
            height = H
            width = W    

            #Output to a single image
            for k in range(num_sets):

                new_im = Image.new('RGBA', (width*(1+mini_batch_size), height))

                curr_image_fname = []
                curr_image_fname.append(image_fname[0])
                for j in range(k*mini_batch_size, (k+1)*mini_batch_size):
                    curr_image_fname.append(image_fname[j])

                images = []

                for fname in curr_image_fname:
                    images.append(Image.open(fname))

                x_offset = 0
                for im in images:
                    new_im.paste(im, (x_offset,0))
                    x_offset += width

                output_image_filename = os.path.join(DUMP_DIR, str(i) + "_" + str(k) + '_collate.png')
                new_im.save(output_image_filename) 

        if i%100==0:
            print("Finished "+str(i)+"/"+str(len(zcache_dataloader)
                )+".")


log_string("=" * 20)
log_string("")
log_string("Num evaluated= "+str(num_evaluated))

print("Done.")



