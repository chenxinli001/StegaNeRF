# Copyright 2021 Alex Yu
# First, install svox2
# Then, python opt.py <path_to>/nerf_synthetic/<scene> -t ckpt/<some_name>
# or use launching script:       sh launch.sh <EXP_NAME> <GPU> <DATA_DIR>

from matplotlib.widgets import EllipseSelector
import torch
import torch.optim
import torch.nn.functional as F
import svox2
import json
import imageio
import os
from os import path
import shutil
import gc
import numpy as np
import math
import argparse
import cv2
from util.dataset import datasets
from util.util import get_expon_lr_func
from util import config_util

from util.util import Timing, get_expon_lr_func, generate_dirs_equirect, viridis_cmap
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import cv2
import pdb

from icecream import ic


from unet import VGG16UNet, Classifier

import lpips

import util.ssim as ssim_utils

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap





def str2bool(str):
    return True if str.lower() == 'true' else False

'''parameter'''
parser = argparse.ArgumentParser()
config_util.define_common_args(parser)

parser.add_argument("--suffix",type=str,default=None)
parser.add_argument("--root",type=str, default=None)
# parser.add_argument("--prefix",type=str, default=None)

'''
sh_mask
'''

# parser.add_argument('--sh_unmask_id_start', type=int, default=-1)
# parser.add_argument('--sh_unmask_id_end', type=int, default=-1)

# parser.add_argument('--unmask_every_color', type=str2bool, default=False)

# parser.add_argument('--sh_unmask_ids', nargs='+', default=[])

parser.add_argument('--sh_unmask_ratio', type=float, default=0.0)
parser.add_argument('--n_power_softw', type=float, default=1.0)

# parser.add_argument('--sh_unmask_ratio_dynamic', action='store_true', default=False)

# parser.add_argument('--unmask_energy_based', action="store_true", default=False) 


# parser.add_argument('--unmask_value_mode', choices=['ranking','energy','energy2'], default='ranking')
# parser.add_argument('--unmask_reweight_mode', choices=['hard','soft'], default='hard')



# parser.add_argument('--dynamic_unmask', action='store_true', default=False)


parser.add_argument("--lr_unet", type=float, default=1e-3)
parser.add_argument("--lr_unet_final", type=float, default=1e-4)
parser.add_argument("--lr_class", type=float, default=1e-4)
parser.add_argument("--lr_class_final", type=float, default=1e-5)

parser.add_argument("--mse_cp", choices=["gt","init"], default="gt")

parser.add_argument("--weight_wm", "--w_wm", type=float, default=1.0)
parser.add_argument("--weight_wm_ctr", "--w_wm_ctr", type=float, default=0.0)
parser.add_argument("--weight_dis_wm", "--w_dis_wm", type=float, default=1.0)

parser.add_argument("--weight_rgb", "--w_rgb", type=float, default=1.0, help='only works for image_mse')

parser.add_argument("--total_epoch", type=int, default=50)

parser.add_argument("--wm_resize", type=int, default=128)


#### Plenoxel parameters
parser.add_argument("--gpuid", type=int, default=2, help='gpu id for cuda')

parser.add_argument("--init_ckpt", type=str, default="", help="initial checkpoint to load")

parser.add_argument("--style", type=str, 
default ='../data/watermarkers/logo.png',
help="path to watermark image")

parser.add_argument("--content_weight", type=float, default=5e-3, help="content loss weight")
parser.add_argument("--img_tv_weight", type=float, default=1, help="image tv loss weight")

parser.add_argument(
    "--vgg_block",
    type=int,
    default=2,
    help="vgg block for extracting feature maps",
)

# parser.add_argument(
#     "--reset_basis_dim",
#     type=int,
#     default=1,
#     help="whether to reset the number of spherical harmonics basis to this specified number",
# )

parser.add_argument(
    "--reset_basis_dim",
    type=int,
    default=0,
    help="whether to reset the number of spherical harmonics basis to this specified number",
)

parser.add_argument(
    "--mse_num_epoches",
    type=int,
    default=2,
    help="epoches for mse loss optimization",
)
parser.add_argument(
    "--nnfm_num_epoches",
    type=int,
    default=10*10,
    help="epoches for running style transfer",
)

#### END of Plenoxel parameters


group = parser.add_argument_group("general")
group.add_argument(
    "--train_dir",
    "-t",
    type=str,
    default="ckpt",
    help="checkpoint and logging directory",
)

group.add_argument(
    "--reso",
    type=str,
    default="[[256, 256, 256], [512, 512, 512]]",
    help="List of grid resolution (will be evaled as json);"
    "resamples to the next one every upsamp_every iters, then "
    + "stays at the last one; "
    + "should be a list where each item is a list of 3 ints or an int",
)

group.add_argument(
    "--upsamp_every",
    type=int,
    default=3 * 12800,
    help="upsample the grid every x iters",
)
group.add_argument("--init_iters", type=int, default=0, help="do not upsample for first x iters")
group.add_argument(
    "--upsample_density_add",
    type=float,
    default=0.0,
    help="add the remaining density by this amount when upsampling",
)

group.add_argument(
    "--basis_type",
    choices=["sh", "3d_texture", "mlp"],
    default="sh",
    help="Basis function type",
)

group.add_argument(
    "--basis_reso",
    type=int,
    default=32,
    help="basis grid resolution (only for learned texture)",
)
group.add_argument("--sh_dim", type=int, default=9, help="SH/learned basis dimensions (at most 10)")

group.add_argument(
    "--mlp_posenc_size",
    type=int,
    default=4,
    help="Positional encoding size if using MLP basis; 0 to disable",
)
group.add_argument("--mlp_width", type=int, default=32, help="MLP width if using MLP basis")

group.add_argument(
    "--background_nlayers",
    type=int,
    default=0,  # 32,
    help="Number of background layers (0=disable BG model)",
)
group.add_argument("--background_reso", type=int, default=512, help="Background resolution")

# poxelnerf通用
group = parser.add_argument_group("optimization")
group.add_argument(
    "--n_iters",
    type=int,
    default=10 * 12800,
    help="total number of iters to optimize for",
)
group.add_argument(
    "--batch_size",
    type=int,
    default=5000,
    # 100000,
    #      2000,
    help="batch size",
)

# TODO: make the lr higher near the end
group.add_argument(
    "--sigma_optim",
    choices=["sgd", "rmsprop"],
    default="rmsprop",
    help="Density optimizer",
)
group.add_argument("--lr_sigma", type=float, default=3e1, help="SGD/rmsprop lr for sigma")
group.add_argument("--lr_sigma_final", type=float, default=5e-2)
group.add_argument("--lr_sigma_decay_steps", type=int, default=250000)
group.add_argument(
    "--lr_sigma_delay_steps",
    type=int,
    default=15000,
    help="Reverse cosine steps (0 means disable)",
)
group.add_argument("--lr_sigma_delay_mult", type=float, default=1e-2)  # 1e-4)#1e-4)


group.add_argument("--sh_optim", choices=["sgd", "rmsprop"], default="rmsprop", help="SH optimizer")
group.add_argument("--lr_sh", type=float, default=1e-2, help="SGD/rmsprop lr for SH")
group.add_argument("--lr_sh_final", type=float, default=5e-6)
group.add_argument("--lr_sh_decay_steps", type=int, default=250000)
group.add_argument(
    "--lr_sh_delay_steps",
    type=int,
    default=0,
    help="Reverse cosine steps (0 means disable)",
)
group.add_argument("--lr_sh_delay_mult", type=float, default=1e-2)

group.add_argument(
    "--lr_fg_begin_step",
    type=int,
    default=0,
    help="Foreground begins training at given step number",
)

# BG LRs
group.add_argument(
    "--bg_optim",
    choices=["sgd", "rmsprop"],
    default="rmsprop",
    help="Background optimizer",
)
group.add_argument("--lr_sigma_bg", type=float, default=3e0, help="SGD/rmsprop lr for background")
group.add_argument(
    "--lr_sigma_bg_final",
    type=float,
    default=3e-3,
    help="SGD/rmsprop lr for background",
)
group.add_argument("--lr_sigma_bg_decay_steps", type=int, default=250000)
group.add_argument(
    "--lr_sigma_bg_delay_steps",
    type=int,
    default=0,
    help="Reverse cosine steps (0 means disable)",
)
group.add_argument("--lr_sigma_bg_delay_mult", type=float, default=1e-2)

group.add_argument("--lr_color_bg", type=float, default=1e-1, help="SGD/rmsprop lr for background")
group.add_argument(
    "--lr_color_bg_final",
    type=float,
    default=5e-6,  # 1e-4,
    help="SGD/rmsprop lr for background",
)
group.add_argument("--lr_color_bg_decay_steps", type=int, default=250000)
group.add_argument(
    "--lr_color_bg_delay_steps",
    type=int,
    default=0,
    help="Reverse cosine steps (0 means disable)",
)
group.add_argument("--lr_color_bg_delay_mult", type=float, default=1e-2)
# END BG LRs

group.add_argument(
    "--basis_optim",
    choices=["sgd", "rmsprop"],
    default="rmsprop",
    help="Learned basis optimizer",
)
# lr_basis  is for SH
group.add_argument("--lr_basis", type=float, default=1e-6, help="SGD/rmsprop lr for SH")  # 2e6,
group.add_argument("--lr_basis_final", type=float, default=1e-6)
group.add_argument("--lr_basis_decay_steps", type=int, default=250000)
group.add_argument(
    "--lr_basis_delay_steps",
    type=int,
    default=0,  # 15000,
    help="Reverse cosine steps (0 means disable)",
)
group.add_argument("--lr_basis_begin_step", type=int, default=0)  # 4 * 12800)
group.add_argument("--lr_basis_delay_mult", type=float, default=1e-2)


group.add_argument("--rms_beta", type=float, default=0.95, help="RMSProp exponential averaging factor")

group.add_argument("--print_every", type=int, default=20, help="print every")
group.add_argument("--save_every", type=int, default=1, help="save every x epochs")
group.add_argument("--eval_every", type=int, default=5, help="evaluate every x epochs")

group.add_argument("--init_sigma", type=float, default=0.1, help="initialization sigma")
group.add_argument("--init_sigma_bg", type=float, default=0.1, help="initialization sigma (for BG)")

# Extra logging
group.add_argument("--log_mse_image", action="store_true", default=False)
group.add_argument("--log_depth_map", action="store_true", default=False)
group.add_argument(
    "--log_depth_map_use_thresh",
    type=float,
    default=None,
    help="If specified, uses the Dex-neRF version of depth with given thresh; else returns expected term",
)


group = parser.add_argument_group("misc experiments")
group.add_argument(
    "--thresh_type",
    choices=["weight", "sigma"],
    default="weight",
    help="Upsample threshold type",
)
group.add_argument(
    "--weight_thresh",
    type=float,
    default=0.0005 * 512,
    #  default=0.025 * 512,
    help="Upsample weight threshold; will be divided by resulting z-resolution",
)
group.add_argument("--density_thresh", type=float, default=5.0, help="Upsample sigma threshold")
group.add_argument(
    "--background_density_thresh",
    type=float,
    default=1.0 + 1e-9,
    help="Background sigma threshold for sparsification",
)
group.add_argument(
    "--max_grid_elements",
    type=int,
    default=44_000_000,
    help="Max items to store after upsampling " "(the number here is given for 22GB memory)",
)

group.add_argument(
    "--tune_mode",
    action="store_true",
    default=False,
    help="hypertuning mode (do not save, for speed)",
)
group.add_argument(
    "--tune_nosave",
    action="store_true",
    default=False,
    help="do not save any checkpoint even at the end",
)


group = parser.add_argument_group("losses")
# Foreground TV
group.add_argument("--lambda_tv", type=float, default=1e-5)
group.add_argument("--tv_sparsity", type=float, default=0.01)
group.add_argument(
    "--tv_logalpha",
    action="store_true",
    default=False,
    help="Use log(1-exp(-delta * sigma)) as in neural volumes",
)

group.add_argument("--lambda_tv_sh", type=float, default=1e-3)
group.add_argument("--tv_sh_sparsity", type=float, default=0.01)

group.add_argument("--lambda_tv_lumisphere", type=float, default=0.0)  # 1e-2)#1e-3)
group.add_argument("--tv_lumisphere_sparsity", type=float, default=0.01)
group.add_argument("--tv_lumisphere_dir_factor", type=float, default=0.0)

group.add_argument("--tv_decay", type=float, default=1.0)

group.add_argument("--lambda_l2_sh", type=float, default=0.0)  # 1e-4)
group.add_argument(
    "--tv_early_only",
    type=int,
    default=1,
    help="Turn off TV regularization after the first split/prune",
)

group.add_argument(
    "--tv_contiguous",
    type=int,
    default=1,
    help="Apply TV only on contiguous link chunks, which is faster",
)
# End Foreground TV

group.add_argument(
    "--lambda_sparsity",
    type=float,
    default=0.0,
    help="Weight for sparsity loss as in SNeRG/PlenOctrees " + "(but applied on the ray)",
)
group.add_argument(
    "--lambda_beta",
    type=float,
    default=0.0,
    help="Weight for beta distribution sparsity loss as in neural volumes",
)


# Background TV
group.add_argument("--lambda_tv_background_sigma", type=float, default=1e-2)
group.add_argument("--lambda_tv_background_color", type=float, default=1e-2)

group.add_argument("--tv_background_sparsity", type=float, default=0.01)
# End Background TV

# Basis TV
group.add_argument(
    "--lambda_tv_basis",
    type=float,
    default=0.0,
    help="Learned basis total variation loss",
)
# End Basis TV

group.add_argument("--weight_decay_sigma", type=float, default=1.0)
group.add_argument("--weight_decay_sh", type=float, default=1.0)

group.add_argument("--lr_decay", action="store_true", default=True)

group.add_argument(
    "--n_train",
    type=int,
    default=None,
    help="Number of training images. Defaults to use all avaiable.",
)

group.add_argument(
    "--nosphereinit",
    action="store_true",
    default=False,
    help="do not start with sphere bounds (please do not use for 360)",
)

args = parser.parse_args()
config_util.maybe_merge_config_file(args)



print('mse_cp is:', args.mse_cp)

os.environ['CUDA_LAUNCH_BLOCKING'] = str(args.gpuid )
device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}')

assert args.lr_sigma_final <= args.lr_sigma, "lr_sigma must be >= lr_sigma_final"
assert args.lr_sh_final <= args.lr_sh, "lr_sh must be >= lr_sh_final"
assert args.lr_basis_final <= args.lr_basis, "lr_basis must be >= lr_basis_final"


scene_name = args.data_dir.split('/')[-2]
dataset_name = args.data_dir.split('/')[-3]

# pdb.set_trace()


style_name = args.style.split('/')[-1]


args.train_dir = f'{args.root}/{dataset_name}-{scene_name}-{style_name}' 


args.train_dir = args.train_dir + '_' + args.suffix

os.makedirs(args.train_dir, exist_ok=True)
summary_writer = SummaryWriter(args.train_dir)

reso_list = json.loads(args.reso)
reso_id = 0

with open(path.join(args.train_dir, "args.json"), "w") as f:
    json.dump(args.__dict__, f, indent=2)
    # Changed name to prevent errors
    shutil.copyfile(__file__, path.join(args.train_dir, "opt_frozen.py"))

# torch.manual_seed(20200823)
# np.random.seed(20200823)

factor = 1

dset_test = datasets[args.dataset_type](
        args.data_dir, split="test", **config_util.build_data_options(args))

dset = datasets[args.dataset_type](
    args.data_dir,
    split="train",
    device=device,
    factor=factor,
    n_images=args.n_train,
    **config_util.build_data_options(args),
)

assert dset.rays.origins.shape == (dset.n_images * dset.h * dset.w, 3)
assert dset.rays.dirs.shape == (dset.n_images * dset.h * dset.w, 3)

if args.background_nlayers > 0 and not dset.should_use_background:
    warn("Using a background model for dataset type " + str(type(dset)) + " which typically does not use background")

# pdb.set_trace()
assert os.path.isfile(args.init_ckpt), "must specify a initial checkpoint"


print(f'reset_basis_dim={args.reset_basis_dim}')
grid = svox2.SparseGrid.load(args.init_ckpt, device=device, reset_basis_dim=args.reset_basis_dim)
grid_init = svox2.SparseGrid.load(args.init_ckpt, device=device, reset_basis_dim=args.reset_basis_dim)
print("Loaded ckpt: ", args.init_ckpt)
print(grid.basis_dim)

'''
Adaptive Gradient Masking
'''
magnitude_dict = {}

# for RGB, 3 channels
for i in range(grid.sh_data.shape[1]//3):
    magnitude_dict[i] = torch.abs(grid.sh_data[:,i]).mean()
    magnitude_dict[i] += torch.abs(grid.sh_data[:,i+grid.sh_data.shape[1]//3]).mean()
    magnitude_dict[i] += torch.abs(grid.sh_data[:,i+grid.sh_data.shape[1]//3*2]).mean()
    magnitude_dict[i] = magnitude_dict[i] / 3

_magnitude_dict = sorted(magnitude_dict.items(), key=lambda d: d[1])


unmask_inds = [ij[0] for ij in _magnitude_dict]
energy_list = [ij[1] for ij in _magnitude_dict]

length = len(unmask_inds)


total_energy = sum(energy_list)
normed_energy = [(1/e).detach() for e in energy_list] #倒数，能量越多动的越少


max_energy = normed_energy[0].detach()

value = [-1]*length

for ij, ind in enumerate(unmask_inds):
    value[ind] = ((normed_energy[ij] / max_energy)**args.n_power_softw).detach()

gradient_reweight_vector = value



optim_basis_mlp = None

if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
    grid.reinit_learned_bases(init_type="sh")


elif grid.basis_type == svox2.BASIS_TYPE_MLP:
    # MLP!
    optim_basis_mlp = torch.optim.Adam(grid.basis_mlp.parameters(), lr=args.lr_basis)



grid.requires_grad_(True)
grid_init.requires_grad_(False)
config_util.setup_render_opts(grid.opt, args)
config_util.setup_render_opts(grid_init.opt, args)
print("Render options", grid.opt)

gstep_id_base = 0
#endregion

# 设置LR衰减
#region
lr_sigma_func = get_expon_lr_func(
    args.lr_sigma,
    args.lr_sigma_final,
    args.lr_sigma_delay_steps,
    args.lr_sigma_delay_mult,
    args.lr_sigma_decay_steps,
)
lr_sh_func = get_expon_lr_func(
    args.lr_sh,
    args.lr_sh_final,
    args.lr_sh_delay_steps,
    args.lr_sh_delay_mult,
    args.lr_sh_decay_steps,
)
lr_basis_func = get_expon_lr_func(
    args.lr_basis,
    args.lr_basis_final,
    args.lr_basis_delay_steps,
    args.lr_basis_delay_mult,
    args.lr_basis_decay_steps,
)
lr_sigma_bg_func = get_expon_lr_func(
    args.lr_sigma_bg,
    args.lr_sigma_bg_final,
    args.lr_sigma_bg_delay_steps,
    args.lr_sigma_bg_delay_mult,
    args.lr_sigma_bg_decay_steps,
)
lr_color_bg_func = get_expon_lr_func(
    args.lr_color_bg,
    args.lr_color_bg_final,
    args.lr_color_bg_delay_steps,
    args.lr_color_bg_delay_mult,
    args.lr_color_bg_decay_steps,
)

lr_unet_func = get_expon_lr_func(
    args.lr_unet,
    args.lr_unet_final,
    0,
    1.0,
    args.total_epoch,
)

lr_class_func = get_expon_lr_func(
    args.lr_class,
    args.lr_class_final,
    0,
    1.0,
    args.total_epoch,
)

# ?
lr_sigma_factor = 1.0
lr_sh_factor = 1.0
lr_basis_factor = 1.0

# lr_unet_factor = 1.0

#endregion

#region
last_upsamp_step = args.init_iters

if args.enable_random:
    warn("Randomness is enabled for training (normal for LLFF & scenes with background)")

###### resize style image such that its long side matches the long side of content images
style_img = imageio.imread(args.style).astype(np.float32) / 255.0
# pdb.set_trace()
# style_h, style_w = style_img.shape[:2]
# content_long_side = max([dset.w, dset.h])

# 将 水印图像设置 为64*64
style_img = cv2.resize(
    style_img,
    (args.wm_resize, args.wm_resize),
    interpolation=cv2.INTER_AREA,
)

imageio.imwrite(
    os.path.join(args.train_dir, "watermark_image.png"),
    np.clip(style_img * 255.0, 0.0, 255.0).astype(np.uint8),
)

style_img = torch.from_numpy(style_img).to(device=device)
print("Watermark image: ", args.style, style_img.shape)

global_start_time = datetime.now()

epoch_id = 0
epoch_size = None
batches_per_epoch = None
batch_size = None


UNET = VGG16UNet(requires_grad=True,args=args).to(device)
Classifier = Classifier(requires_grad=True).to(device)

''' optimizer for unet'''
optimizer_unet = torch.optim.Adam(params=UNET.parameters(), lr=args.lr_unet)
optimizer_class = torch.optim.Adam(params=Classifier.parameters(), lr=args.lr_class)


'''metric setting for ssim and lpips'''

lpips_alex = lpips.LPIPS(net='alex') # best forward scores
lpips_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

def ssim(img1, img2, window_size = 11, size_average = True, format='NCHW'):
    if format == 'HWC':
        img1 = img1.permute([2, 0, 1])[None, ...]
        img2 = img2.permute([2, 0, 1])[None, ...]
    elif format == 'NHWC':
        img1 = img1.permute([0, 3, 1, 2])
        img2 = img2.permute([0, 3, 1, 2])

    return ssim_utils.ssim(img1, img2, window_size, size_average)

def lpips(img1, img2, net='alex', format='NCHW'):
    # pdb.set_trace()

    if format == 'HWC':
        # pdb.set_trace()
        img1 = img1.permute([2, 0, 1])[None, ...]
        img2 = img2.permute([2, 0, 1])[None, ...]
    elif format == 'NHWC':
        img1 = img1.permute([0, 3, 1, 2])
        img2 = img2.permute([0, 3, 1, 2])

    if net == 'alex':
        return lpips_alex(img1, img2)
    elif net == 'vgg':
        return lpips_vgg(img1, img2)

def to8b(x):
    return  (255*(x-x.min())/(x.max()-x.min())).astype(np.uint8)


while True:

    def eval_step():
        # Put in a function to avoid memory leak
        print('Eval step')
        with torch.no_grad():
            stats_test_cp_gt = {'psnr' : 0.0, 'mse' : 0.0, 'ssim':0.0, 'lpips':0.0}
            stats_test_cp_init = {'psnr' : 0.0, 'mse' : 0.0, 'ssim':0.0, 'lpips':0.0}
            stats_test_init = {'psnr' : 0.0, 'mse' : 0.0, 'ssim':0.0, 'lpips':0.0}
            wm_stats_test = {'psnr' : 0.0, 'mse' : 0.0, 'mse_ctr':0.0, 'ssim':0.0, 'lpips':0.0, 'dis_wm_acc':0.0}

            # Standard set
            N_IMGS_TO_EVAL = min(20 if epoch_id > 0 else 5, dset_test.n_images)
            N_IMGS_TO_SAVE = N_IMGS_TO_EVAL # if not args.tune_mode else 1
            img_eval_interval = dset_test.n_images // N_IMGS_TO_EVAL
            img_save_interval = (N_IMGS_TO_EVAL // N_IMGS_TO_SAVE) #1
            img_ids = range(0, dset_test.n_images, img_eval_interval)

            # Special 'very hard' specular + fuzz set
            #  img_ids = [2, 5, 7, 9, 21,
            #             44, 45, 47, 49, 56,
            #             80, 88, 99, 115, 120,
            #             154]
            #  img_save_interval = 1

            n_images_gen = 0
            for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
                c2w = dset_test.c2w[img_id].to(device=device)
                cam = svox2.Camera(c2w,
                                   dset_test.intrins.get('fx', img_id),
                                   dset_test.intrins.get('fy', img_id),
                                   dset_test.intrins.get('cx', img_id),
                                   dset_test.intrins.get('cy', img_id),
                                   width=dset_test.get_image_size(img_id)[1],
                                   height=dset_test.get_image_size(img_id)[0],
                                   ndc_coeffs=dset_test.ndc_coeffs)

                # Full-img rendering                  
                rgb_pred_test = grid.volume_render_image(cam, use_kernel=True)
                rgb_gt_test = dset_test.gt[img_id].to(device=device)

                all_mses = ((rgb_gt_test - rgb_pred_test) ** 2).cpu()

                rgb_pred_test_init = grid_init.volume_render_image(cam, use_kernel=True)

                all_mses_cp_init = ((rgb_pred_test_init - rgb_pred_test) ** 2).cpu()

                all_mses_init = ( (rgb_pred_test_init - rgb_gt_test) **2).cpu()

                
                rgb_gt_test = rgb_gt_test.permute(2, 0, 1).unsqueeze(0).contiguous() # 1,3,h,w
                rgb_pred_test = rgb_pred_test.permute(2, 0, 1).unsqueeze(0).contiguous() # 1,3,h,w
                rgb_pred_test_init = rgb_pred_test_init.permute(2, 0, 1).unsqueeze(0).contiguous() # 1,3,h,w
                
                '''TEST: obtain nerf and decoder output'''
                UNET.eval()
                Classifier.eval()

                bs = rgb_gt_test.shape[0]
                out_class = Classifier(torch.cat((rgb_pred_test,rgb_pred_test_init.detach()),0))

                out_c, out_c_ctr = out_class[:bs], out_class[bs:]

                dis_wm_acc = ( torch.sum(out_c>=0.5) + torch.sum(out_c_ctr<0.5) ) / ( torch.sum(torch.ones_like(out_c)) + torch.sum(torch.ones_like(out_c_ctr)) ) 

                out_unet = UNET( torch.cat( (rgb_pred_test,rgb_pred_test_init.detach() ), 0), torch.cat( (out_c,out_c_ctr), 0), return_c=False,cmask=False)
                
                out_wm, out_wm_ctr = out_unet[:bs], out_unet[bs:]

                wm_mses = ((out_wm - style_img.permute(2,0,1).unsqueeze(0)) ** 2).cpu()
                wm_ctr_mses = ( (out_wm_ctr - torch.zeros_like(out_wm_ctr)) **2).cpu()


                if i % img_save_interval == 0:

                    summary_writer.add_image(f'test/gt_{img_id:04d}', rgb_gt_test.squeeze(0).permute(1,2,0).cpu().clamp_max_(1.0), global_step=gstep_id_base, dataformats='HWC')

                    summary_writer.add_image(f'test/pred_init_{img_id:04d}', rgb_pred_test_init.squeeze(0).permute(1,2,0).cpu().clamp_max_(1.0), global_step=gstep_id_base, dataformats='HWC')

                    summary_writer.add_image(f'test/pred_{img_id:04d}', rgb_pred_test.squeeze(0).permute(1,2,0).cpu().clamp_max_(1.0), global_step=gstep_id_base, dataformats='HWC')

                    summary_writer.add_image(f'test/wm_{img_id:04d}',to8b(out_wm.squeeze(0).permute(1,2,0).cpu().detach().numpy()), global_step=gstep_id_base, dataformats='HWC')

                    summary_writer.add_image(f'test/wm_ctr_{img_id:04d}', to8b(out_wm_ctr.squeeze(0).permute(1,2,0).cpu().detach().numpy()), global_step=gstep_id_base, dataformats='HWC')

                    summary_writer.add_image(f'test/style_{img_id:04d}', style_img.cpu().clamp_max_(1.0), global_step=gstep_id_base, dataformats='HWC')


                    summary_writer.add_image(f'test/res_cp_init_{img_id:04d}', viridis_cmap(  torch.abs(rgb_pred_test-rgb_pred_test_init).squeeze(0).permute(1,2,0).mean(-1,keepdim=True).cpu().detach().numpy() ), global_step=gstep_id_base, dataformats='HWC')

                    summary_writer.add_image(f'test/res_cp_gt_{img_id:04d}', viridis_cmap( torch.abs(rgb_pred_test-rgb_gt_test).squeeze(0).permute(1,2,0).mean(-1,keepdim=True).cpu().detach().numpy()), global_step=gstep_id_base, dataformats='HWC')


                    if args.log_mse_image:
                        mse_img = all_mses / all_mses.max()
                        summary_writer.add_image(f'test/mse_map_{img_id:04d}',
                                mse_img, global_step=gstep_id_base, dataformats='HWC')
                    if args.log_depth_map:
                        depth_img = grid.volume_render_depth_image(cam,
                                    args.log_depth_map_use_thresh if
                                    args.log_depth_map_use_thresh else None
                                )
                        depth_img = viridis_cmap(depth_img.cpu())
                        summary_writer.add_image(f'test/depth_map_{img_id:04d}',
                                depth_img,
                                global_step=gstep_id_base, dataformats='HWC')


                ''' FOR NERF RECON, compared with GT'''
                mse_num : float = all_mses.mean().item()
                # pdb.set_trace()
                psnr = -10.0 * math.log10(mse_num)
                if math.isnan(psnr):
                    print('NAN PSNR', i, img_id, mse_num)
                    assert False
                stats_test_cp_gt['mse'] += mse_num
                stats_test_cp_gt['psnr'] += psnr
                stats_test_cp_gt['ssim'] += ssim(rgb_pred_test.squeeze(0).permute(1,2,0), rgb_gt_test.squeeze(0).permute(1,2,0), format='HWC')
                stats_test_cp_gt['lpips'] += lpips(rgb_pred_test.squeeze(0).permute(1,2,0).cpu(), rgb_gt_test.squeeze(0).permute(1,2,0).cpu(), format='HWC')

                ''' FOR NERF RECON, compared with INIT NERF OUTPUT'''
                mse_num : float = all_mses_cp_init.mean().item()
                if mse_num == 0: 
                    psnr = 0
                else:
                    psnr = -10.0 * math.log10(mse_num)
                if math.isnan(psnr):
                    print('NAN PSNR', i, img_id, mse_num)
                    assert False
                stats_test_cp_init['mse'] += mse_num
                stats_test_cp_init['psnr'] += psnr
                stats_test_cp_init['ssim'] += ssim(rgb_pred_test.squeeze(0).permute(1,2,0), rgb_pred_test_init.squeeze(0).permute(1,2,0), format='HWC')
                stats_test_cp_init['lpips'] += lpips(rgb_pred_test.squeeze(0).permute(1,2,0).cpu(), rgb_pred_test_init.squeeze(0).permute(1,2,0).cpu(), format='HWC')

                ''' FOR INIT NERF RECON, compared with GT'''
                mse_num : float = all_mses_init.mean().item()
                if mse_num == 0: 
                    psnr = 0
                else:
                    psnr = -10.0 * math.log10(mse_num)
                if math.isnan(psnr):
                    print('NAN PSNR', i, img_id, mse_num)
                    assert False
                stats_test_init['mse'] += mse_num
                stats_test_init['psnr'] += psnr
                stats_test_init['ssim'] += ssim(rgb_pred_test_init.squeeze(0).permute(1,2,0), rgb_gt_test.squeeze(0).permute(1,2,0), format='HWC')
                stats_test_init['lpips'] += lpips(rgb_pred_test_init.squeeze(0).permute(1,2,0).cpu(), rgb_gt_test.squeeze(0).permute(1,2,0).cpu(), format='HWC')

                
                '''FOR WM RECON'''
                mse_num : float = wm_mses.mean().item()
                psnr = -10.0 * math.log10(mse_num)
                if math.isnan(psnr):
                    print('NAN PSNR', i, img_id, mse_num)
                    assert False
                wm_stats_test['mse'] += mse_num

                wm_stats_test['mse_ctr'] += wm_ctr_mses.mean().item()

                wm_stats_test['psnr'] += psnr
                wm_stats_test['ssim'] += ssim(out_wm.squeeze(0).permute(1,2,0), style_img,  format='HWC')
                wm_stats_test['lpips'] += lpips(out_wm.squeeze(0).permute(1,2,0).cpu(), style_img.cpu(), format='HWC')
                
                wm_stats_test['dis_wm_acc'] += dis_wm_acc.item()

                rgb_pred_test = rgb_gt_test = None

                n_images_gen += 1

            # 应该用不上，应该是SH
            if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE or grid.basis_type == svox2.BASIS_TYPE_MLP:
                 # Add spherical map visualization
                EQ_RESO = 256
                eq_dirs = generate_dirs_equirect(EQ_RESO * 2, EQ_RESO)
                eq_dirs = torch.from_numpy(eq_dirs).to(device=device).view(-1, 3)

                if grid.basis_type == svox2.BASIS_TYPE_MLP:
                    sphfuncs = grid._eval_basis_mlp(eq_dirs)
                else:
                    sphfuncs = grid._eval_learned_bases(eq_dirs)
                sphfuncs = sphfuncs.view(EQ_RESO, EQ_RESO*2, -1).permute([2, 0, 1]).cpu().numpy()

                stats = [(sphfunc.min(), sphfunc.mean(), sphfunc.max())
                        for sphfunc in sphfuncs]
                sphfuncs_cmapped = [viridis_cmap(sphfunc) for sphfunc in sphfuncs]
                for im, (minv, meanv, maxv) in zip(sphfuncs_cmapped, stats):
                    cv2.putText(im, f"{minv=:.4f} {meanv=:.4f} {maxv=:.4f}", (10, 20),
                                0, 0.5, [255, 0, 0])
                sphfuncs_cmapped = np.concatenate(sphfuncs_cmapped, axis=0)
                summary_writer.add_image(f'test/spheric',
                        sphfuncs_cmapped, global_step=gstep_id_base, dataformats='HWC')
                # END add spherical map visualization

            for stat_name in stats_test_cp_gt:
                stats_test_cp_gt[stat_name] /= n_images_gen
                summary_writer.add_scalar('test/rgb_cp_gt/' + stat_name, stats_test_cp_gt[stat_name], global_step=gstep_id_base)
            
            for stat_name in stats_test_cp_init:
                stats_test_cp_init[stat_name] /= n_images_gen
                summary_writer.add_scalar('test/rgb_cp_init/' + stat_name, stats_test_cp_init[stat_name], global_step=gstep_id_base)
            
            for stat_name in stats_test_init:
                stats_test_init[stat_name] /= n_images_gen
                summary_writer.add_scalar('test/rgb_init/' + stat_name, stats_test_init[stat_name], global_step=gstep_id_base)

            for stat_name in wm_stats_test:
                wm_stats_test[stat_name] /= n_images_gen
                summary_writer.add_scalar('test/wm/' + stat_name, wm_stats_test[stat_name], global_step=gstep_id_base)

            summary_writer.add_scalar('epoch_id', float(epoch_id), global_step=gstep_id_base)

    '''Call Eval '''
    if epoch_id % max(factor, args.eval_every) == 0: #and (epoch_id > 0 or not args.tune_mode):
        # NOTE: we do an eval sanity check, if not in tune_mode
        eval_step()
        gc.collect()


    def train_step(optim_type):
        print("Training epoch: ", epoch_id, epoch_size, batches_per_epoch, batch_size, optim_type)
        pbar = tqdm(enumerate(range(0, epoch_size, batch_size)), total=batches_per_epoch)
        for iter_id, batch_begin in pbar:
            stats = {}
            wm_status = {}

            gstep_id = iter_id + gstep_id_base


            lr_sigma = lr_sigma_func(gstep_id) * lr_sigma_factor
            lr_sh = lr_sh_func(gstep_id) * lr_sh_factor

            lr_basis = lr_basis_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
            lr_sigma_bg = lr_sigma_bg_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
            lr_color_bg = lr_color_bg_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
            
            # for epoch_id
            lr_unet = lr_unet_func(epoch_id)
            lr_class = lr_class_func(epoch_id) # fix bug

            # lr = lr_unet_func(gstep_id) * lr_unet_factor

            '''MSE LOSS'''
            if optim_type == "ray_mse":
                batch_end = min(batch_begin + args.batch_size, epoch_size)
                batch_origins = dset.rays.origins[batch_begin:batch_end].to(device)
                batch_dirs = dset.rays.dirs[batch_begin:batch_end].to(device)

                rgb_gt = dset.rays.gt[batch_begin:batch_end].to(device)

                rays = svox2.Rays(batch_origins, batch_dirs)

                rgb_pred = grid.volume_render_fused(
                    rays,
                    rgb_gt,
                    beta_loss=args.lambda_beta,
                    sparsity_loss=args.lambda_sparsity,
                    randomize=args.enable_random,
                    is_rgb_gt=True,
                    reset_grad_indexers=True,
                )
                mse = F.mse_loss(rgb_gt, rgb_pred)
                # Stats
                psnr = -10.0 * math.log10(mse.detach().item())
                stats["mse_loss"] = mse
                stats["psnr"] = torch.tensor(psnr)
                
                # stats['ssim'] = ssim(rgb_pred.squeeze(0).permute(1,2,0), rgb_gt.squeeze(0).permute(1,2,0), format='HWC')
                # stats['lpips'] = lpips(rgb_pred.squeeze(0).permute(1,2,0).cpu(), rgb_gt.squeeze(0).permute(1,2,0).cpu(), format='HWC')


            '''IMAGE STYLE LOSS'''
            if optim_type == "image_wm":
        
                num_views, view_height, view_width = dset.n_images, dset.h, dset.w
       
                img_id = np.random.randint(low=0, high=num_views)
                rays = svox2.Rays(
                    dset.rays.origins.view(num_views, view_height * view_width, 3)[img_id].to(device),
                    dset.rays.dirs.view(num_views, view_height * view_width, 3)[img_id].to(device),
                )
                # 29,756*1008,3 > 756*1008,3
                def compute_image_loss():
                    #   Stop the gradient to forward a full image
                    with torch.no_grad():
                        cam = svox2.Camera(
                            dset.c2w[img_id].to(device=device),
                            dset.intrins.get("fx", img_id),
                            dset.intrins.get("fy", img_id),
                            dset.intrins.get("cx", img_id),
                            dset.intrins.get("cy", img_id),
                            width=view_width,
                            height=view_height,
                            ndc_coeffs=dset.ndc_coeffs,
                        )
         
                        rgb_pred = grid.volume_render_image(cam, use_kernel=True)
                        rgb_pred_init = grid_init.volume_render_image(cam, use_kernel=True)
                        rgb_gt = dset.rays.gt.view(num_views, view_height, view_width, 3)[img_id].to(
                            device
                        )
                        rgb_gt = rgb_gt.permute(2, 0, 1).unsqueeze(0).contiguous() # 1,3,h,w
                        rgb_pred = rgb_pred.permute(2, 0, 1).unsqueeze(0).contiguous() # 1,3,h,w
                        rgb_pred_init = rgb_pred_init.permute(2, 0, 1).unsqueeze(0).contiguous() # 1,3,h,w

                    rgb_pred.requires_grad_(True)
                    rgb_pred_init.requires_grad_(False)
                    
                    UNET.train()
                    Classifier.train()


                    bs = rgb_pred.shape[0]
    
                    out_class = Classifier(torch.cat((rgb_pred,rgb_pred_init.detach()),0))

                    out_c, out_c_ctr = out_class[:bs], out_class[bs:]

                    dis_wm_loss = F.binary_cross_entropy(out_class, torch.stack( [torch.ones(bs), torch.zeros(bs)]).to(device) )

                    dis_wm_acc = ( torch.sum(out_c>=0.5) + torch.sum(out_c_ctr<0.5) ) / ( torch.sum(torch.ones_like(out_c)) + torch.sum(torch.ones_like(out_c_ctr)) ) 

                    '''
                    2022-10-12 DOUBLE CHECK: out_class.detach() exists!
                    '''

                    out_unet = UNET( torch.cat( (rgb_pred,rgb_pred_init.detach() ), 0), torch.cat( (out_c,out_c_ctr), 0).detach(), return_c=False,cmask=False)
                    bs = rgb_gt.shape[0]
                    out_wm, out_wm_ctr = out_unet[:bs], out_unet[bs:]

                    
                    wm_loss = F.mse_loss(out_wm, style_img.permute(2,0,1).unsqueeze(0))
                    wm_ctr_loss = F.mse_loss(out_wm_ctr, torch.zeros_like(out_wm_ctr))

                    loss_dict = {}
                    loss_dict['wm_loss'] = wm_loss
                    loss_dict['wm_ctr_loss'] = wm_ctr_loss
                    loss_dict['dis_wm_loss'] = dis_wm_loss
                    loss_dict['dis_wm_acc'] = dis_wm_acc

                    cp_gt_mse_loss = F.mse_loss(rgb_gt, rgb_pred)
                    cp_init_mse_loss = F.mse_loss(rgb_pred_init.detach(), rgb_pred)
                    init_mse_loss = F.mse_loss(rgb_pred_init.detach(), rgb_gt)

                    if args.mse_cp == 'gt':
                        mse_loss = cp_gt_mse_loss
                    elif args.mse_cp == 'init':
                        mse_loss = cp_init_mse_loss

                    loss_dict['mse_loss'] = mse_loss

                    loss_dict['cp_gt_mse_loss'] = cp_gt_mse_loss
                    loss_dict['cp_init_mse_loss'] = cp_init_mse_loss
                    
                    loss_dict['init_mse_loss'] = init_mse_loss
                    
                    # pdb.set_trace()
                    if loss_dict['mse_loss'].item() == 0:
                        psnr = 0
                    else:
                        psnr = -10.0 * math.log10(loss_dict['mse_loss'])
                    loss_dict["psnr"] = torch.tensor(psnr)
                    loss_dict['ssim'] = ssim(rgb_pred.squeeze(0).permute(1,2,0), rgb_gt.squeeze(0).permute(1,2,0), format='HWC')
                    loss_dict['lpips'] = lpips(rgb_pred.squeeze(0).permute(1,2,0).cpu(), rgb_gt.squeeze(0).permute(1,2,0).cpu(), format='HWC') 


                    # print('Weight:', args.weight_rgb, args.weight_wm, args.weight_wm_ctr, args.weight_dis_wm)
                    # print(mse_loss, wm_loss)

                    loss = args.weight_rgb*mse_loss + args.weight_wm*wm_loss + args.weight_wm_ctr*wm_ctr_loss + args.weight_dis_wm*dis_wm_loss

                    loss = loss.mean()

                    optimizer_class.param_groups[0]['lr'] = lr_class
                    optimizer_unet.param_groups[0]['lr'] = lr_unet
                

                    optimizer_class.zero_grad()
                    optimizer_unet.zero_grad()
                    loss.backward()
                    optimizer_class.step()
                    optimizer_unet.step()
                    
                    # Select IDs FOR VIEWING
                    if img_id in [0,1,2]:
                    
                        summary_writer.add_image(f'train/style_image_{img_id:04d}',
                                to8b(style_img.cpu().detach().numpy()), global_step=gstep_id_base, dataformats='HWC')

                        summary_writer.add_image(f'train/wm_image_{img_id:04d}',
                                to8b(out_wm.squeeze(0).permute(1,2,0).cpu().detach().numpy()), global_step=gstep_id_base, dataformats='HWC')
                        
                        summary_writer.add_image(f'train/wm_ctr_image_{img_id:04d}',
                                to8b(out_wm_ctr.squeeze(0).permute(1,2,0).cpu().detach().numpy()), global_step=gstep_id_base, dataformats='HWC')


                        summary_writer.add_image(f'train/gt_{img_id:04d}', rgb_gt.squeeze(0).permute(1,2,0).cpu().clamp_max_(1.0), global_step=gstep_id_base, dataformats='HWC')

                        summary_writer.add_image(f'train/pred_{img_id:04d}', rgb_pred.squeeze(0).permute(1,2,0).cpu().clamp_max_(1.0), global_step=gstep_id_base, dataformats='HWC')

                        summary_writer.add_image(f'train/pred_init_{img_id:04d}', rgb_pred_init.squeeze(0).permute(1,2,0).cpu().clamp_max_(1.0), global_step=gstep_id_base, dataformats='HWC')

                        # 
                        # viridis = mpl.colormaps['viridis']
                        # pdb.set_trace()
                        # depth_img = viridis_cmap(depth_img.cpu())
                        # res
                        summary_writer.add_image(f'train/res_cp_init_{img_id:04d}',
                            viridis_cmap( torch.abs(rgb_pred-rgb_pred_init).squeeze(0).permute(1,2,0).mean(-1,keepdim=True).cpu().detach().numpy() ), global_step=gstep_id_base, dataformats='HWC')
                        summary_writer.add_image(f'train/res_cp_gt_{img_id:04d}',
                            viridis_cmap( torch.abs(rgb_pred-rgb_gt).squeeze(0).permute(1,2,0).mean(-1,keepdim=True).cpu().detach().numpy()), global_step=gstep_id_base, dataformats='HWC')

   
                    rgb_pred_grad = rgb_pred.grad.squeeze(0).permute(1, 2, 0).contiguous().clone().detach().view(-1, 3)
        
                    return rgb_pred_grad, loss_dict

    
                rgb_pred_grad, loss_dict = compute_image_loss()
                rgb_pred = []
        
                grid.alloc_grad_indexers()

         
                for view_batch_start in range(0, view_height * view_width, args.batch_size):
                    rgb_pred_patch = grid.volume_render_fused(
                        rays[view_batch_start : view_batch_start + args.batch_size],
                        rgb_pred_grad[view_batch_start : view_batch_start + args.batch_size],
                        beta_loss=args.lambda_beta,
                        sparsity_loss=args.lambda_sparsity,
                        randomize=args.enable_random,
                        is_rgb_gt=False,
                        reset_grad_indexers=False,
                    )
                    rgb_pred.append(rgb_pred_patch.clone().detach())
                rgb_pred = torch.cat(rgb_pred, dim=0).reshape(view_height, view_width, 3)

                # Stats
                for x in loss_dict:
                    stats[x] = loss_dict[x].item()


            if (iter_id + 1) % args.print_every == 0:
                log_str = ""
                for stat_name in stats:
                    summary_writer.add_scalar('train/'+stat_name, stats[stat_name], global_step=gstep_id)
                    log_str += "{:.4f} ".format(stats[stat_name])
                pbar.set_description(f"{gstep_id} {log_str}")

                summary_writer.add_scalar("lr/lr_sh", lr_sh, global_step=gstep_id)
                summary_writer.add_scalar("lr/lr_sigma", lr_sigma, global_step=gstep_id)

                summary_writer.add_scalar("lr/lr_unet", lr_unet, global_step=gstep_id)
                summary_writer.add_scalar("lr/lr_class", lr_class, global_step=gstep_id)

                if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                    summary_writer.add_scalar("lr/lr_basis", lr_basis, global_step=gstep_id)
                if grid.use_background:
                    summary_writer.add_scalar("lr/lr_sigma_bg", lr_sigma_bg, global_step=gstep_id)
                    summary_writer.add_scalar("lr/lr_color_bg", lr_color_bg, global_step=gstep_id)
                


            if args.weight_decay_sh < 1.0:
                grid.sh_data.data *= args.weight_decay_sigma
            if args.weight_decay_sigma < 1.0:
                grid.density_data.data *= args.weight_decay_sh

            # Apply TV/Sparsity regularizers
            if args.lambda_tv > 0.0:
                #  with Timing("tv_inpl"):
                grid.inplace_tv_grad(
                    grid.density_data.grad,
                    scaling=args.lambda_tv,
                    sparse_frac=args.tv_sparsity,
                    logalpha=args.tv_logalpha,
                    ndc_coeffs=dset.ndc_coeffs,
                    contiguous=args.tv_contiguous,
                )
            if args.lambda_tv_sh > 0.0:
                #  with Timing("tv_color_inpl"):
                grid.inplace_tv_color_grad(
                    grid.sh_data.grad,
                    scaling=args.lambda_tv_sh,
                    sparse_frac=args.tv_sh_sparsity,
                    ndc_coeffs=dset.ndc_coeffs,
                    contiguous=args.tv_contiguous,
                )
            if args.lambda_tv_lumisphere > 0.0:
                grid.inplace_tv_lumisphere_grad(
                    grid.sh_data.grad,
                    scaling=args.lambda_tv_lumisphere,
                    dir_factor=args.tv_lumisphere_dir_factor,
                    sparse_frac=args.tv_lumisphere_sparsity,
                    ndc_coeffs=dset.ndc_coeffs,
                )
            if args.lambda_l2_sh > 0.0:
                grid.inplace_l2_color_grad(grid.sh_data.grad, scaling=args.lambda_l2_sh)
            if grid.use_background and (args.lambda_tv_background_sigma > 0.0 or args.lambda_tv_background_color > 0.0):
                grid.inplace_tv_background_grad(
                    grid.background_data.grad,
                    scaling=args.lambda_tv_background_color,
                    scaling_density=args.lambda_tv_background_sigma,
                    sparse_frac=args.tv_background_sparsity,
                    contiguous=args.tv_contiguous,
                )
            if args.lambda_tv_basis > 0.0:
                tv_basis = grid.tv_basis()
                loss_tv_basis = tv_basis * args.lambda_tv_basis
                loss_tv_basis.backward()



            '''
            Perform Adaptive Gradient Masking
            '''

            _sh_data_grad = grid.sh_data.grad.detach().clone()

            for i in range(9):
                grid.sh_data.grad[:,i] = _sh_data_grad[:,i]*gradient_reweight_vector[i]
                grid.sh_data.grad[:,i+9] = _sh_data_grad[:,i+9]*gradient_reweight_vector[i]
                grid.sh_data.grad[:,i+18] = _sh_data_grad[:,i+18]*gradient_reweight_vector[i]


            # Manual SGD/rmsprop step
            # print(lr_sigma)
            if lr_sigma > 0.0:
                grid.optim_density_step(lr_sigma, beta=args.rms_beta, optim=args.sigma_optim)
            grid.optim_sh_step(lr_sh, beta=args.rms_beta, optim=args.sh_optim)

            if grid.use_background:
                grid.optim_background_step(lr_sigma_bg, lr_color_bg, beta=args.rms_beta, optim=args.bg_optim)

            if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                grid.optim_basis_step(lr_basis, beta=args.rms_beta, optim=args.basis_optim)
            elif grid.basis_type == svox2.BASIS_TYPE_MLP:
                optim_basis_mlp.step()
                optim_basis_mlp.zero_grad()


    # img_id = np.random.randint(low=0, high=dset.n_images)
    img_id = 0

    cam = svox2.Camera(
        dset.c2w[img_id].to(device=device),
        dset.intrins.get("fx", img_id),
        dset.intrins.get("fy", img_id),
        dset.intrins.get("cx", img_id),
        dset.intrins.get("cy", img_id),
        width=dset.get_image_size(img_id)[1],
        height=dset.get_image_size(img_id)[0],
        ndc_coeffs=dset.ndc_coeffs,
    )

    rgb_pred = grid.volume_render_image(cam, use_kernel=True).detach().cpu().numpy()
    print('value range of image:',rgb_pred.min(),rgb_pred.max())
    imageio.imwrite(
        os.path.join(args.train_dir, f"logim_{epoch_id}.png"),
        np.clip(rgb_pred * 255.0, 0.0, 255.0).astype(np.uint8),
    )
    

    # if args.mse_mode == 'ray':
    #     if epoch_id%args.mse_every==0 and epoch_id!=0:
    #         epoch_size = dset.rays.origins.size(0)
    #         batch_size = args.batch_size
    #         batches_per_epoch = (epoch_size - 1) // batch_size + 1
    #         train_step(optim_type="ray_mse")
    #     else:
    #         epoch_size = dset.n_images
    #         batch_size = 1
    #         batches_per_epoch = (dset.n_images - 1) // batch_size + 1
    #         train_step(optim_type="image_wm")
        
    # elif args.mse_mode == 'image':
    #     epoch_size = dset.n_images
    #     batch_size = 1
    #     batches_per_epoch = (dset.n_images - 1) // batch_size + 1
    #     train_step(optim_type="image_wm+mse")

    # epoch_size = dset.rays.origins.size(0)
    # batch_size = args.batch_size
    # batches_per_epoch = (epoch_size - 1) // batch_size + 1
    # train_step(optim_type="ray_mse")

    epoch_size = dset.n_images
    batch_size = 1
    batches_per_epoch = (dset.n_images - 1) // batch_size + 1
    train_step(optim_type="image_wm")


    epoch_id += 1
    gstep_id_base += batches_per_epoch
    torch.cuda.empty_cache()
    gc.collect()

    if epoch_id >= args.total_epoch:

        global_stop_time = datetime.now()
        secs = (global_stop_time - global_start_time).total_seconds()
        timings_file = open(os.path.join(args.train_dir, "time_mins.txt"), "w")
        timings_file.write(f"{secs / 60}\n")
        timings_file.close()

        # style实验的 npz
        ckpt_path = path.join(args.train_dir, "ckpt.npz")
        grid.save(ckpt_path)

        decoder_state_dict = {
        'epoch':epoch_id,
        'UNET_state_dict': UNET.state_dict(),
        'Classifier_state_dict': Classifier.state_dict()
        }
        torch.save(decoder_state_dict, path.join(args.train_dir, "decoder.pth"))

        img_id = np.random.randint(low=0, high=dset.n_images)
        cam = svox2.Camera(
            dset.c2w[img_id].to(device=device),
            dset.intrins.get("fx", img_id),
            dset.intrins.get("fy", img_id),
            dset.intrins.get("cx", img_id),
            dset.intrins.get("cy", img_id),
            width=dset.get_image_size(img_id)[1],
            height=dset.get_image_size(img_id)[0],
            ndc_coeffs=dset.ndc_coeffs,
        )
        rgb_pred = grid.volume_render_image(cam, use_kernel=True)
        rgb_pred = rgb_pred.detach().cpu().numpy()

        imageio.imwrite(
            os.path.join(args.train_dir, f"logim_{epoch_id}_final.png"),
            np.clip(rgb_pred * 255.0, 0.0, 255.0).astype(np.uint8),
        )

        break
