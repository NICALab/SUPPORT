## To train model, run below.
import os
import torch
import random
import numpy as np
import time
from colab.utils import random_transform, gen_train_dataloader, DatasetSUPPORT_test_stitch
from colab.model import SUPPORT
from tqdm import tqdm


def train_SUPPORT(img, model_size=4, n_epochs=20):
    random.seed(0)
    torch.manual_seed(0)

    # ----------
    # Initialize: Create sample and checkpoint directories
    # ----------
    cuda = torch.cuda.is_available()
    rng = np.random.default_rng(0)

    #-----------
    # Dataset
    # ----------
    dataloader_train = gen_train_dataloader([21, 128, 128], [1, 64, 64], 64, img)
    print(len(dataloader_train))

    # ----------
    # Model, Optimizers, and Loss
    # ----------
    # model = SUPPORT(in_channels=21, mid_channels=[16, 32, 64, 128, 256], depth=5,\
    #      blind_conv_channels=64, one_by_one_channels=[32, 16],\
    #             last_layer_channels=[64, 32, 16], bs_size=[1, 19])
    model = SUPPORT(in_channels=21,
                    mid_channels=[4*model_size,
                                  8*model_size,
                                  16*model_size,
                                  32*model_size,
                                  64*model_size],
                    depth=5,
                    blind_conv_channels=16*model_size,
                    one_by_one_channels=[8*model_size, 4*model_size],
                    last_layer_channels=[16*model_size,
                                         8*model_size,
                                         4*model_size],
                    bs_size=[1, 19])

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    if cuda:
        model = model.cuda()

    # ----------
    # Training & Validation
    # ----------
    i = 0
    
    for epoch in range(n_epochs):
        # initialize
        model.train()

        L1_pixelwise = torch.nn.L1Loss()
        L2_pixelwise = torch.nn.MSELoss()

        # training
        for _, data in tqdm(enumerate(dataloader_train)):

            (noisy_image, _, _) = data
            _, T, _, _ = noisy_image.shape
            noisy_image, _ = random_transform(noisy_image, None, rng, False)
            
            if cuda:
                noisy_image = noisy_image.cuda()
            noisy_image_target = torch.unsqueeze(noisy_image[:, int(T/2), :, :], dim=1)

            noisy_image_denoised = model(noisy_image)
            loss_l1_pixelwise = L1_pixelwise(noisy_image_denoised, noisy_image_target)
            loss_l2_pixelwise = L2_pixelwise(noisy_image_denoised, noisy_image_target)
            loss_sum = 0.5 * loss_l1_pixelwise + 0.5 * loss_l2_pixelwise

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

            i += 1

            if i == 10:
                torch.cuda.synchronize()
                start = time.perf_counter()
            if i == 110:
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                ETA = (elapsed / 100) * (len(dataloader_train) * n_epochs - 100)
                print(f"ETA : {ETA:.4f}s")

            if i % 2000 == 0:
                torch.save(model.state_dict(), f"./model_ep{epoch}.pt")

    return model


def trainshort_SUPPORT(img, pretrain_path="", model_size=4, n_epochs=3):
    random.seed(0)
    torch.manual_seed(0)

    # ----------
    # Initialize: Create sample and checkpoint directories
    # ----------
    cuda = torch.cuda.is_available()
    rng = np.random.default_rng(0)

    #-----------
    # Dataset
    # ----------
    dataloader_train = gen_train_dataloader([21, 128, 128], [1, 64, 64], 64, img)
    print(len(dataloader_train))

    # ----------
    # Model, Optimizers, and Loss
    # ----------
    model = SUPPORT(in_channels=21,
                    mid_channels=[4*model_size,
                                  8*model_size,
                                  16*model_size,
                                  32*model_size,
                                  64*model_size],
                    depth=5,
                    blind_conv_channels=16*model_size,
                    one_by_one_channels=[8*model_size, 4*model_size],
                    last_layer_channels=[16*model_size,
                                         8*model_size,
                                         4*model_size],
                    bs_size=[1, 19])
    model.load_state_dict(torch.load(pretrain_path))

    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if cuda:
        model = model.cuda()

    # ----------
    # Training & Validation
    # ----------
    i = 0
    
    for epoch in range(n_epochs):
        # initialize
        model.train()

        L1_pixelwise = torch.nn.L1Loss()
        L2_pixelwise = torch.nn.MSELoss()

        # training
        for _, data in tqdm(enumerate(dataloader_train)):

            (noisy_image, _, _) = data
            _, T, _, _ = noisy_image.shape
            noisy_image, _ = random_transform(noisy_image, None, rng, False)
            
            if cuda:
                noisy_image = noisy_image.cuda()
            noisy_image_target = torch.unsqueeze(noisy_image[:, int(T/2), :, :], dim=1)

            noisy_image_denoised = model(noisy_image)
            loss_l1_pixelwise = L1_pixelwise(noisy_image_denoised, noisy_image_target)
            loss_l2_pixelwise = L2_pixelwise(noisy_image_denoised, noisy_image_target)
            loss_sum = 0.5 * loss_l1_pixelwise + 0.5 * loss_l2_pixelwise

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

            i += 1

            if i == 10:
                torch.cuda.synchronize()
                start = time.perf_counter()
            if i == 110:
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                ETA = (elapsed / 100) * (len(dataloader_train) * n_epochs - 100)
                print(f"ETA : {ETA:.4f}s")

            if i % 2000 == 0:
                torch.save(model.state_dict(), f"./model_ep{epoch}.pt")

    return model


@torch.no_grad()
def test_SUPPORT(img, model_param):
    cuda = torch.cuda.is_available()

    model_size = int(model_param["model_size"].item())
    model = SUPPORT(in_channels=21,
                    mid_channels=[4*model_size,
                                  8*model_size,
                                  16*model_size,
                                  32*model_size,
                                  64*model_size],
                    depth=5,
                    blind_conv_channels=16*model_size,
                    one_by_one_channels=[8*model_size, 4*model_size],
                    last_layer_channels=[16*model_size,
                                         8*model_size,
                                         4*model_size],
                    bs_size=[1, 19])
    if cuda:
        model = model.cuda()

    model.load_state_dict(model_param)
    model.eval()

    testset = DatasetSUPPORT_test_stitch(img, patch_size=[21, 128, 128],\
        patch_interval=[1, 64, 64])
    testloader = torch.utils.data.DataLoader(testset, batch_size=16)

    denoised_stack = np.zeros(testloader.dataset.noisy_image.shape, dtype=np.float32)
    
    print(len(testloader))
    for _, (noisy_image, _, single_coordinate) in tqdm(enumerate(testloader)):
        noisy_image = noisy_image.cuda() #[b, z, y, x]
        noisy_image_denoised = model(noisy_image)
        T = noisy_image.size(1)
        for bi in range(noisy_image.size(0)): 
            stack_start_w = int(single_coordinate['stack_start_w'][bi])
            stack_end_w = int(single_coordinate['stack_end_w'][bi])
            patch_start_w = int(single_coordinate['patch_start_w'][bi])
            patch_end_w = int(single_coordinate['patch_end_w'][bi])

            stack_start_h = int(single_coordinate['stack_start_h'][bi])
            stack_end_h = int(single_coordinate['stack_end_h'][bi])
            patch_start_h = int(single_coordinate['patch_start_h'][bi])
            patch_end_h = int(single_coordinate['patch_end_h'][bi])

            stack_start_s = int(single_coordinate['init_s'][bi])
            
            denoised_stack[stack_start_s+(T//2), stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                = noisy_image_denoised[bi].squeeze()[patch_start_h:patch_end_h, patch_start_w:patch_end_w].cpu()

    # change nan values to 0 and denormalize
    denoised_stack = denoised_stack * testloader.dataset.std_image.numpy() + testloader.dataset.mean_image.numpy()
    # denoised_stack = np.clip(denoised_stack*255, a_min=0, a_max=255*255).astype(np.uint16)

    return denoised_stack


