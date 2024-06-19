import numpy as np
import torch
import skimage.io as skio

from tqdm import tqdm
from src.utils.dataset import DatasetSUPPORT_test_stitch
from model.SUPPORT import SUPPORT


def validate(test_dataloader, model):
    """
    Validate a model with a test data
    
    Arguments:
        test_dataloader: (Pytorch DataLoader)
            Should be DatasetFRECTAL_test_stitch!
        model: (Pytorch nn.Module)

    Returns:
        denoised_stack: denoised image stack (Numpy array with dimension [T, X, Y])
    """
    with torch.no_grad():
        model.eval()
        # initialize denoised stack to NaN array.
        denoised_stack = np.zeros(test_dataloader.dataset.noisy_image.shape, dtype=np.float32)
        
        # stitching denoised stack
        # insert the results if the stack value was NaN
        # or, half of the output volume
        for _, (noisy_image, _, single_coordinate) in enumerate(tqdm(test_dataloader, desc="validate")):
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
        denoised_stack = denoised_stack * test_dataloader.dataset.std_image.numpy() + test_dataloader.dataset.mean_image.numpy()

        return denoised_stack


if __name__ == '__main__':
    ########## Change it with your data ##############
    data_file = "./data/Xiaohan/sample/cropped2-2.tif"
    model_file = "./src/GUI/trained_models/bs1.pth" # "./results/saved_models/mytest/model_0.pth"
    output_file = "./results/xiaohan_denoised_0.tif"
    patch_size = [61, 64, 64]
    patch_interval = [1, 32, 32]
    batch_size = 16    # lower it if memory exceeds.
    bs_size = 3    # modify if you changed bs_size when training.
    bp_mode = False
    ##################################################

    model = SUPPORT(in_channels=61, mid_channels=[16, 32, 64, 128, 256], depth=5,\
            blind_conv_channels=64, one_by_one_channels=[32, 16], last_layer_channels=[64, 32, 16], bs_size=bs_size, bp=bp_mode).cuda()

    model.load_state_dict(torch.load(model_file))

    demo_tif = torch.from_numpy(skio.imread(data_file).astype(np.float32)).type(torch.FloatTensor)
    demo_tif = demo_tif[:, :, :]

    testset = DatasetSUPPORT_test_stitch(demo_tif, patch_size=patch_size,\
        patch_interval=patch_interval)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    denoised_stack = validate(testloader, model)

    print(denoised_stack.shape)
    skio.imsave(output_file, denoised_stack[(model.in_channels-1)//2:-(model.in_channels-1)//2, : , :], metadata={'axes': 'TYX'})
