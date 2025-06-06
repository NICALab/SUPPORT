import os
import math
import argparse
import numpy as np
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument("--random_seed", type=int, default=0, help="random seed for rng")
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from (need epoch-1 model)")
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--exp_name", type=str, default="myEXP", help="name of the experiment")
    parser.add_argument("--results_dir", type=str, default="./results", help="root directory to save results")
    parser.add_argument("--input_frames", type=int, default=61, help="# of input frames")
    # parser.add_argument("--cuda_device", type=int, default=[0], nargs="+", help="cuda devices to use")

    # dataset
    parser.add_argument("--is_zarr", action="store_true", help="noisy_data is zarr")
    parser.add_argument("--is_folder", action="store_true", help="noisy_data is folder")
    parser.add_argument("--noisy_data", type=str, nargs="+", help="List of path to the noisy data")
    parser.add_argument("--patch_size", type=int, default=[61, 128, 128], nargs="+", help="size of the patches")
    parser.add_argument("--patch_interval", type=int, default=[1, 64, 64], nargs="+", help="size of the patch interval")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")

    # model
    parser.add_argument("--depth", type=int, default=5, help="the number of blind spot convolutions, must be an odd number")
    parser.add_argument("--blind_conv_channels", type=int, default=64, help="the number of channels of blind spot convolutions")
    parser.add_argument("--one_by_one_channels", type=int, default=[32, 16], nargs="+", help="the number of channels of 1x1 convolutions")
    parser.add_argument("--last_layer_channels", type=int, default=[64, 32, 16], nargs="+", help="the number of channels of 1x1 convs after UNet")
    parser.add_argument("--bs_size", type=int, default=[3, 3], nargs="+", help="the size of the blind spot")
    parser.add_argument("--bp", action="store_true", help="blind plane")
    parser.add_argument("--unet_channels", type=int, default=[64, 128, 256, 512, 1024], nargs="+", help="the number of channels of UNet")

    # training
    parser.add_argument("--lr", type=float, default=5e-4, help="adam: learning rate")
    parser.add_argument("--loss_coef", type=float, default=[0.5, 0.5], nargs="+", help="L1/L2 loss coefficients")

    # util
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision for training")
    parser.add_argument("--use_CPU", action="store_true", help="use CPU")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="number of batches to prefetch")
    parser.add_argument("--logging_interval_batch", type=int, default=50, help="interval between logging info (in batches)")
    parser.add_argument("--logging_interval", type=int, default=1, help="interval between logging info (in epochs)")
    parser.add_argument("--sample_interval", type=int, default=10, help="interval between saving denoised samples")
    parser.add_argument("--sample_max_t", type=int, default=600, help="maximum time step of saving sample")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving trained models (in epochs)")
    parser.add_argument("--checkpoint_interval_batch", type=int, default=10000, help="interval between saving trained models (in batches)")
    opt = parser.parse_args()

    # argument checking
    if (opt.input_frames) != opt.patch_size[0]:
        raise Exception("input frames must be equal to z-frames of patch_size")
    if len(opt.loss_coef) != 2:
        raise Exception("loss_coef must be length-2 array")

    if not opt.is_zarr:
        if opt.is_folder:
            all_files = []

            for i in opt.noisy_data:
                all_files += sorted([str(p) for p in Path(i).rglob('*') if p.is_file()])

            opt.noisy_data = all_files
    else:
        if opt.is_folder:
            all_dirs = []
            for folder in opt.noisy_data:
                for root, dirs, files in os.walk(folder):
                    for d in dirs:
                        if d.endswith(".zarr"):
                            all_dirs.append(os.path.join(root, d))
                    dirs[:] = [d for d in dirs if not d.endswith(".zarr")]
            opt.noisy_data = sorted(all_dirs)

    # print the noisy files
    print("Noisy files:")
    for i in opt.noisy_data:
        print(i)

    return opt


def get_coordinate(img_size, patch_size, patch_interval):
    """DeepCAD version of stitching
    https://github.com/cabooster/DeepCAD/blob/53a9b8491170e298aa7740a4656b4f679ded6f41/DeepCAD_pytorch/data_process.py#L374
    """
    whole_s, whole_h, whole_w = img_size
    img_s, img_h, img_w = patch_size
    gap_s, gap_h, gap_w = patch_interval

    cut_w = (img_w - gap_w)/2
    cut_h = (img_h - gap_h)/2
    cut_s = (img_s - gap_s)/2

    # print(whole_s, whole_h, whole_w)
    # print(img_s, img_h, img_w)
    # print(gap_s, gap_h, gap_w)

    num_w = math.ceil((whole_w-img_w+gap_w)/gap_w)
    num_h = math.ceil((whole_h-img_h+gap_h)/gap_h)
    num_s = math.ceil((whole_s-img_s+gap_s)/gap_s)

    coordinate_list = []
    for x in range(0,num_h):
        for y in range(0,num_w):
            for z in range(0,num_s):
                single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0, 'init_s':0, 'end_s':0}
                if x != (num_h-1):
                    init_h = gap_h*x
                    end_h = gap_h*x + img_h
                elif x == (num_h-1):
                    init_h = whole_h - img_h
                    end_h = whole_h

                if y != (num_w-1):
                    init_w = gap_w*y
                    end_w = gap_w*y + img_w
                elif y == (num_w-1):
                    init_w = whole_w - img_w
                    end_w = whole_w

                if z != (num_s-1):
                    init_s = gap_s*z
                    end_s = gap_s*z + img_s
                elif z == (num_s-1):
                    init_s = whole_s - img_s
                    end_s = whole_s
                single_coordinate['init_h'] = init_h
                single_coordinate['end_h'] = end_h
                single_coordinate['init_w'] = init_w
                single_coordinate['end_w'] = end_w
                single_coordinate['init_s'] = init_s
                single_coordinate['end_s'] = end_s

                if y == 0:
                    if num_w > 1:
                        single_coordinate['stack_start_w'] = y*gap_w
                        single_coordinate['stack_end_w'] = y*gap_w+img_w-cut_w
                        single_coordinate['patch_start_w'] = 0
                        single_coordinate['patch_end_w'] = img_w-cut_w
                    else:
                        single_coordinate['stack_start_w'] = 0
                        single_coordinate['stack_end_w'] = img_w
                        single_coordinate['patch_start_w'] = 0
                        single_coordinate['patch_end_w'] = img_w
                elif y == num_w-1:
                    single_coordinate['stack_start_w'] = whole_w-img_w+cut_w
                    single_coordinate['stack_end_w'] = whole_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = img_w
                else:
                    single_coordinate['stack_start_w'] = y*gap_w+cut_w
                    single_coordinate['stack_end_w'] = y*gap_w+img_w-cut_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = img_w-cut_w

                if x == 0:
                    if num_h > 1:
                        single_coordinate['stack_start_h'] = x*gap_h
                        single_coordinate['stack_end_h'] = x*gap_h+img_h-cut_h
                        single_coordinate['patch_start_h'] = 0
                        single_coordinate['patch_end_h'] = img_h-cut_h
                    else:
                        single_coordinate['stack_start_h'] = 0
                        single_coordinate['stack_end_h'] = x*gap_h+img_h
                        single_coordinate['patch_start_h'] = 0
                        single_coordinate['patch_end_h'] = img_h
                elif x == num_h-1:
                    single_coordinate['stack_start_h'] = whole_h-img_h+cut_h
                    single_coordinate['stack_end_h'] = whole_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = img_h
                else:
                    single_coordinate['stack_start_h'] = x*gap_h+cut_h
                    single_coordinate['stack_end_h'] = x*gap_h+img_h-cut_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = img_h-cut_h

                if z == 0:
                    if num_s > 1:
                        single_coordinate['stack_start_s'] = z*gap_s
                        single_coordinate['stack_end_s'] = z*gap_s+img_s-cut_s
                        single_coordinate['patch_start_s'] = 0
                        single_coordinate['patch_end_s'] = img_s-cut_s
                    else:
                        single_coordinate['stack_start_s'] = z*gap_s
                        single_coordinate['stack_end_s'] = z*gap_s+img_s
                        single_coordinate['patch_start_s'] = 0
                        single_coordinate['patch_end_s'] = img_s
                elif z == num_s-1:
                    single_coordinate['stack_start_s'] = whole_s-img_s+cut_s
                    single_coordinate['stack_end_s'] = whole_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = img_s
                else:
                    single_coordinate['stack_start_s'] = z*gap_s+cut_s
                    single_coordinate['stack_end_s'] = z*gap_s+img_s-cut_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = img_s-cut_s

                coordinate_list.append(single_coordinate)

    return coordinate_list
