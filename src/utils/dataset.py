import skimage.io as skio
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from src.utils.util import get_coordinate


def random_transform(input, target, rng, is_rotate=True):
    """
    Randomly rotate/flip the image

    Arguments:
        input: input image stack (Pytorch Tensor with dimension [b, T, X, Y])
        target: targer image stack (Pytorch Tensor with dimension [b, T, X, Y]), can be None
        rng: numpy random number generator
    
    Returns:
        input: randomly rotated/flipped input image stack (Pytorch Tensor with dimension [b, T, X, Y])
        target: randomly rotated/flipped target image stack (Pytorch Tensor with dimension [b, T, X, Y])
    """
    rand_num = rng.integers(0, 4) # random number for rotation
    rand_num_2 = rng.integers(0, 2) # random number for flip

    if is_rotate:
        if rand_num == 1:
            input = torch.rot90(input, k=1, dims=(2, 3))
            if target is not None:
                target = torch.rot90(target, k=1, dims=(2, 3))
        elif rand_num == 2:
            input = torch.rot90(input, k=2, dims=(2, 3))
            if target is not None:
                target = torch.rot90(target, k=2, dims=(2, 3))
        elif rand_num == 3:
            input = torch.rot90(input, k=3, dims=(2, 3))
            if target is not None:
                target = torch.rot90(target, k=3, dims=(2, 3))
    
    if rand_num_2 == 1:
        input = torch.flip(input, dims=[2])
        if target is not None:
            target = torch.flip(target, dims=[2])

    return input, target


def normalize(image):
    """
    Normalize the image to [mean/std]=[0/1]

    Arguments:
        image: image stack (Pytorch Tensor with dimension [T, X, Y])

    Returns:
        image: normalized image stack (Pytorch Tensor with dimension [T, X, Y])
        mean_image: mean of the image stack (np.float)
        std_image: standard deviation of the image stack (np.float)
    """
    mean_image = torch.mean(image)
    std_image = torch.std(image)

    image -= mean_image
    image /= std_image

    return image, mean_image, std_image


class DatasetSUPPORT(Dataset):
    def __init__(self, noisy_images, patch_size=[61, 128, 128], patch_interval=[10, 64, 64], load_to_memory=True,\
        transform=None, random_patch=True, random_patch_seed=0):
        """
        Arguments:
            noisy_images: list of noisy image stack ([Tensor with dimension [t, x, y]])
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

        # initialize
        self.data_weight = []
        for noisy_image in noisy_images:
            self.data_weight.append(torch.numel(noisy_image))

        self.patch_size = patch_size
        self.patch_interval = patch_interval
        self.transform = transform
        self.random_patch = random_patch
        self.patch_rng = np.random.default_rng(random_patch_seed)

        self.noisy_images = noisy_images
        self.mean_images = []
        self.std_images = []
        for idx, noisy_image in enumerate(noisy_images):
            noisy_image, mean_image, std_image = normalize(noisy_image)
            self.noisy_images[idx] = noisy_image
            self.mean_images.append(mean_image)
            self.std_images.append(std_image)
        self.mean_images = torch.tensor(self.mean_images)
        self.std_images = torch.tensor(self.std_images)

        # generate index
        self.indices_ds = []
        for noisy_image in self.noisy_images:
            indices = []
            tmp_size = noisy_image.size()
            if np.any(tmp_size < np.array(self.patch_size)):
                raise Exception("patch size is larger than data size")

            for k in range(3):
                z_range = list(range(0, tmp_size[k]-self.patch_size[k]+1, self.patch_interval[k]))
                if tmp_size[k] - self.patch_size[k] > z_range[-1]:
                    z_range.append(tmp_size[k]-self.patch_size[k])
                indices.append(z_range)
            self.indices_ds.append(indices)


    def __len__(self):
        total = 0
        for indices in self.indices_ds:
            total += len(indices[0]) * len(indices[1]) * len(indices[2])

        return total

    def __getitem__(self, i):
        # slicing
        if self.random_patch:
            ds_idx = self.patch_rng.choice(len(self.data_weight), 1)[0]
            t_idx = self.patch_rng.integers(0, self.noisy_images[ds_idx].size()[0]-self.patch_size[0]+1)
            y_idx = self.patch_rng.integers(0, self.noisy_images[ds_idx].size()[1]-self.patch_size[1]+1)
            z_idx = self.patch_rng.integers(0, self.noisy_images[ds_idx].size()[2]-self.patch_size[2]+1)
        else:
            ds_idx = 0
            t_idx = self.indices_ds[ds_idx][0][i // (len(self.indices_ds[ds_idx][1]) * len(self.indices_ds[ds_idx][2]))]
            y_idx = self.indices_ds[ds_idx][1][(i % (len(self.indices_ds[ds_idx][1]) * len(self.indices_ds[ds_idx][2]))) // len(self.indices_ds[ds_idx][2])]
            z_idx = self.indices_ds[ds_idx][2][i % len(self.indices_ds[ds_idx][2])]

        # input dataset range
        t_range = slice(t_idx, t_idx + self.patch_size[0])
        y_range = slice(y_idx, y_idx + self.patch_size[1])
        z_range = slice(z_idx, z_idx + self.patch_size[2])
        
        noisy_image = self.noisy_images[ds_idx][t_range, y_range, z_range]
        
        return noisy_image, torch.tensor([[t_idx, t_idx + self.patch_size[0]],\
            [y_idx, y_idx + self.patch_size[1]], [z_idx, z_idx + self.patch_size[2]]]), torch.tensor(ds_idx)


class DatasetSUPPORT_test_stitch(Dataset):
    def __init__(self, noisy_image, patch_size=[61, 128, 128], patch_interval=[10, 64, 64], load_to_memory=True,\
        transform=None, random_patch=False, random_patch_seed=0):
        """
        Arguments:
            noisy_image: noisy image stack (Tensor with dimension [t, x, y])
            patch_size: size of the patch ([int]), ([t, x, y])
            patch_interval: interval between each patch ([int]), ([t, x, y])
            load_to_memory: whether load data into memory or not (bool)
            transform: function of transformation (function)
            random_patch: sample patch in random or not (bool)
            random_patch_seed: seed for randomness (int)
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
        self.noisy_image, self.mean_image, self.std_image = normalize(self.noisy_image)

        # generate index
        self.indices = []
        tmp_size = self.noisy_image.size()
        if np.any(tmp_size < np.array(self.patch_size)):
            raise Exception("patch size is larger than data size")

        self.indices = get_coordinate(tmp_size, patch_size, patch_interval)            

    def __len__(self):
        return len(self.indices) # len(self.indices[0]) * len(self.indices[1]) * len(self.indices[2])

    def __getitem__(self, i):
        # slicing
        if self.random_patch:
            idx = self.patch_rng.integers(0, len(self.indices) - 1)
        else:
            idx = i
        single_coordinate = self.indices[idx]
        
        # input dataset range
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']

        # for stitching dataset range
        noisy_image = self.noisy_image[init_s:end_s,init_h:end_h,init_w:end_w]

        # transform
        if self.transform:
            rand_i = self.patch_rng.integers(0, self.transform.n_masks)
            rand_t = self.patch_rng.integers(0, 2)
            noisy_image = self.transform.mask(noisy_image, rand_i, rand_t)

        return noisy_image, torch.empty(1), single_coordinate


def gen_train_dataloader(patch_size, patch_interval, batch_size, noisy_data_list):
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
        noisy_image = torch.from_numpy(skio.imread(noisy_data).astype(np.float32)).type(torch.FloatTensor)
        print(f"Loaded {noisy_data} Shape : {noisy_image.shape}")
        if len(noisy_image.shape) == 2:
            noisy_image = noisy_image.unsqueeze(0)
        T, _, _ = noisy_image.shape
        noisy_images_train.append(noisy_image)

    dataset_train = DatasetSUPPORT(noisy_images_train, patch_size=patch_size,\
        patch_interval=patch_interval, transform=None, random_patch=True)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    
    return dataloader_train
