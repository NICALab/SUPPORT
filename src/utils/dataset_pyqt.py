import skimage.io as skio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils.util import get_coordinate
from src.utils.dataset import DatasetSUPPORT


def normalize(image, mean_image=None, std_image=None):
    """
    Normalize the image to [mean/std]=[0/1]

    Arguments:
        image: image stack (Pytorch Tensor with dimension [T, X, Y])

    Returns:
        image: normalized image stack (Pytorch Tensor with dimension [T, X, Y])
        mean_image: mean of the image stack (np.float)
        std_image: standard deviation of the image stack (np.float)
    """
    if mean_image is None or std_image is None:
        mean_image = torch.mean(image)
        std_image = torch.std(image)

    image -= mean_image
    image /= std_image

    return image, mean_image, std_image


class DatasetSupport_test_stitch(Dataset):
    def __init__(
        self,
        noisy_image,
        patch_size=[61, 128, 128],
        patch_interval=[10, 64, 64],
        load_to_memory=True,
        transform=None,
        random_patch=False,
        random_patch_seed=0,
        mean_image=None,
        std_image=None,
    ):
        """
        Arguments:
            noisy_image: noisy image stack (Tensor with dimension [t, x, y])
            clean_image: clean image stack (Tensor with dimension [t, x, y]), default: None
            patch_size: size of the patch ([int]), ([t, x, y])
            patch_interval: interval between each patch ([int]), ([t, x, y])
            load_to_memory: whether load data into memory or not (bool)
            transform: function of transformation (function)
            random_patch: sample patch in random or not (bool)
            random_patch_seed: seed for randomness (int)
            algorithm: the algorithm of use (str)
        """
        # check arguments
        if len(patch_size) != 3:
            raise Exception("length of patch_size must be 3")
        if len(patch_interval) != 3:
            raise Exception("length of patch_interval must be 3")

        self.patch_size = patch_size
        self.patch_interval = patch_interval
        self.transform = transform
        self.random_patch = random_patch
        self.patch_rng = np.random.default_rng(random_patch_seed)
        self.noisy_image = noisy_image
        self.noisy_image, self.mean_image, self.std_image = normalize(
            self.noisy_image, mean_image, std_image
        )

        # generate index
        self.indices = []
        tmp_size = self.noisy_image.size()
        if np.any(tmp_size < np.array(self.patch_size)):
            raise Exception("patch size is larger than data size")

        self.indices = get_coordinate(tmp_size, patch_size, patch_interval)

    def __len__(self):
        return len(
            self.indices
        )

    def __getitem__(self, i):
        # slicing
        if self.random_patch:
            idx = self.patch_rng.integers(0, len(self.indices) - 1)
        else:
            idx = i
        single_coordinate = self.indices[idx]

        # input dataset range
        init_h = single_coordinate["init_h"]
        end_h = single_coordinate["end_h"]
        init_w = single_coordinate["init_w"]
        end_w = single_coordinate["end_w"]
        init_s = single_coordinate["init_s"]
        end_s = single_coordinate["end_s"]

        # for stitching dataset range
        noisy_image = self.noisy_image[init_s:end_s, init_h:end_h, init_w:end_w]

        # transform
        if self.transform:
            rand_i = self.patch_rng.integers(0, self.transform.n_masks)
            rand_t = self.patch_rng.integers(0, 2)
            noisy_image = self.transform.mask(noisy_image, rand_i, rand_t)

        return noisy_image, torch.empty(1), single_coordinate


def gen_train_dataloader(
    patch_size,
    patch_interval,
    batch_size,
    noisy_data_list
):
    """
    Generate dataloader for training

    Arguments:
        patch_size: opt.patch_size
        patch_interval: opt.patch_interval
        noisy_data_list: opt.noisy_data

    Returns:
        dataloader_train
    """
    noisy_images_train = []

    for noisy_data in noisy_data_list:
        noisy_image = torch.from_numpy(skio.imread(noisy_data).astype(np.float32)).type(
            torch.FloatTensor
        )
        T, _, _ = noisy_image.shape
        noisy_images_train.append(noisy_image)

    dataset_train = DatasetSUPPORT(
        noisy_images_train,
        patch_size=patch_size,
        patch_interval=patch_interval,
        transform=None,
        random_patch=True,
    )
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    return dataloader_train
