import os
import random
import logging
import time
import numpy as np
import torch
import skimage.io as skio

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.utils.dataset import gen_train_dataloader, random_transform
from src.utils.util import parse_arguments
from model.SUPPORT import SUPPORT


def train(train_dataloader, model, optimizer, rng, writer, epoch, opt):
    """
    Train a model for a single epoch

    Arguments:
        train_dataloader: (Pytorch DataLoader)
        model: (Pytorch nn.Module)
        optimizer: (Pytorch optimzer)
        rng: numpy random number generator
        writer: (Tensorboard writer)
        epoch: epoch of training (int)
        opt: argparse dictionary

    Returns:
        loss_list: list of total loss of each batch ([float])
        loss_list_l1: list of L1 loss of each batch ([float])
        loss_list_l2: list of L2 loss of each batch ([float])
        corr_list: list of correlation of each batch ([float])
    """

    is_rotate = True if model.bs_size[0] == model.bs_size[1] else False
    
    # initialize
    model.train()
    loss_list_l1 = []
    loss_list_l2 = []
    loss_list = []

    L1_pixelwise = torch.nn.L1Loss()
    L2_pixelwise = torch.nn.MSELoss()

    loss_coef = opt.loss_coef

    # training
    for i, data in enumerate(tqdm(train_dataloader)):

        (noisy_image, _, ds_idx) = data
        noisy_image, _ = random_transform(noisy_image, None, rng, is_rotate)
        
        B, T, X, Y = noisy_image.shape
        noisy_image = noisy_image.cuda()
        noisy_image_target = torch.unsqueeze(noisy_image[:, int(T/2), :, :], dim=1)

        optimizer.zero_grad()
        noisy_image_denoised = model(noisy_image)
        loss_l1_pixelwise = L1_pixelwise(noisy_image_denoised, noisy_image_target)
        loss_l2_pixelwise = L2_pixelwise(noisy_image_denoised, noisy_image_target)
        loss_sum = loss_coef[0] * loss_l1_pixelwise + loss_coef[1] * loss_l2_pixelwise
        loss_sum.backward()
        optimizer.step()

        loss_list_l1.append(loss_l1_pixelwise.item())
        loss_list_l2.append(loss_l2_pixelwise.item())
        loss_list.append(loss_sum.item())

        # print log
        if (epoch % opt.logging_interval == 0) and (i % opt.logging_interval_batch == 0):
            loss_mean = np.mean(np.array(loss_list))
            loss_mean_l1 = np.mean(np.array(loss_list_l1))
            loss_mean_l2 = np.mean(np.array(loss_list_l2))

            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            writer.add_scalar("Loss_l1/train_batch", loss_mean_l1, epoch*len(train_dataloader) + i)
            writer.add_scalar("Loss_l2/train_batch", loss_mean_l2, epoch*len(train_dataloader) + i)
            writer.add_scalar("Loss/train_batch", loss_mean, epoch*len(train_dataloader) + i)
            
            logging.info(f"[{ts}] Epoch [{epoch}/{opt.n_epochs}] Batch [{i+1}/{len(train_dataloader)}] "+\
                f"loss : {loss_mean:.4f}, loss_l1 : {loss_mean_l1:.4f}, loss_l2 : {loss_mean_l2:.4f}")

    return loss_list, loss_list_l1, loss_list_l2


if __name__=="__main__":
    random.seed(0)
    torch.manual_seed(0)

    # ----------
    # Initialize: Create sample and checkpoint directories
    # ----------
    opt = parse_arguments()
    cuda = torch.cuda.is_available() and (not opt.use_CPU)
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    rng = np.random.default_rng(opt.random_seed)

    os.makedirs(opt.results_dir + "/images/{}".format(opt.exp_name), exist_ok=True)
    os.makedirs(opt.results_dir + "/saved_models/{}".format(opt.exp_name), exist_ok=True)
    os.makedirs(opt.results_dir + "/logs".format(opt.exp_name), exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=opt.results_dir + "/logs/{}.log".format(opt.exp_name),\
        filemode="a", format="%(name)s - %(levelname)s - %(message)s")
    writer = SummaryWriter(opt.results_dir + "/tsboard/{}".format(opt.exp_name))

    #-----------
    # Dataset
    # ----------
    dataloader_train = gen_train_dataloader(opt.patch_size, opt.patch_interval, opt.batch_size, \
        opt.noisy_data)

    # ----------
    # Model, Optimizers, and Loss
    # ----------
    model = SUPPORT(in_channels=opt.input_frames, mid_channels=opt.unet_channels, depth=opt.depth,\
         blind_conv_channels=opt.blind_conv_channels, one_by_one_channels=opt.one_by_one_channels,\
                last_layer_channels=opt.last_layer_channels, bs_size=opt.bs_size, bp=opt.bp)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    if cuda:
        model = model.cuda()
    
    if opt.epoch != 0:
        model.load_state_dict(torch.load(opt.results_dir + "/saved_models/%s/model_%d.pth" % (opt.exp_name, opt.epoch-1)))
        optimizer.load_state_dict(torch.load(opt.results_dir + "/saved_models/%s/optimizer_%d.pth" % (opt.exp_name, opt.epoch-1)))
        print('Loaded pre-trained model and optimizer weights of epoch {}'.format(opt.epoch-1))

    # ----------
    # Training & Validation
    # ----------
    for epoch in range(opt.epoch, opt.n_epochs):
        loss_list, loss_list_l1, loss_list_l2 =\
            train(dataloader_train, model, optimizer, rng, writer, epoch, opt)

        # logging
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        if (epoch % opt.logging_interval == 0):
            loss_mean = np.mean(np.array(loss_list))
            loss_mean_l1 = np.mean(np.array(loss_list_l1))
            loss_mean_l2 = np.mean(np.array(loss_list_l2))

            writer.add_scalar("Loss/train", loss_mean, epoch)
            writer.add_scalar("Loss_l1/train", loss_mean_l1, epoch)
            writer.add_scalar("Loss_l2/train", loss_mean_l2, epoch)
            logging.info(f"[{ts}] Epoch [{epoch}/{opt.n_epochs}] "+\
                f"loss : {loss_mean:.4f}, loss_l1 : {loss_mean_l1:.4f}, loss_l2 : {loss_mean_l2:.4f}")
        
        if (opt.checkpoint_interval != -1) and (epoch % opt.checkpoint_interval == 0):
            torch.save(model.state_dict(), opt.results_dir + "/saved_models/%s/model_%d.pth" % (opt.exp_name, epoch))
            torch.save(optimizer.state_dict(), opt.results_dir + "/saved_models/%s/optimizer_%d.pth" % (opt.exp_name, epoch))

        # if (epoch % opt.sample_interval == 0):
        #     skio.imsave(opt.results_dir + "/images/%s/denoised_%d.pth" % (opt.exp_name, epoch), )
            
