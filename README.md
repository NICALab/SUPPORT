# SUPPORT: statistically unbiased prediction enables accurate denoising of voltage imaging data

<div align="center"><img src="thumbnail.png" width="80%"/></div>

## About SUPPORT
Here we present SUPPORT (**S**tatistically **U**nbiased **P**rediction utilizing s**P**ati**O**tempo**R**al information in imaging da**T**a), **a self-supervised denoising method** for **voltage imaging data**. SUPPORT is based on the insight that a pixel value in voltage imaging data is highly dependent on its spatially neighboring pixels in the same time frame even when its temporally adjacent frames do not provide useful information for statistical prediction. Such spatiotemporal dependency is captured and utilized to accurately denoise voltage imaging data in which the existence of the action potential in a time frame cannot be inferred by the information in other frames. We show, through simulation and experiments, that SUPPORT enables precise denoising of voltage imaging data while preserving the underlying dynamics in the scene. 

We also show that SUPPORT can be used for denoising **time-lapse fluorescence microscopy images** of Caenorhabditis elegans (C. elegans), in which the imaging speed is not faster than the locomotion of the worm, as well as **static volumetric images** of Penicillium and mouse embryos. SUPPORT is exceptionally compelling for denoising voltage imaging and time-lapse imaging data, and is even effective for denoising **calcium imaging data**.

For more details, please see the accompanying research publication "Statistically unbiased prediction enables accurate denoising of voltage imaging data".

## Installation
1. Clone the repository
```
git clone git@github.com:NICALab/SUPPORT.git
```

2. Navigate into the cloned folder
```
cd ./SUPPORT
```

3. Create the conda environment
```
conda env create -f env.yml
```

4. Activate the conda environment
```
conda activate SUPPORT
```

5. Install Pytorch with **the version compatible with your OS and platform** from https://pytorch.org/get-started/locally/
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

## Getting Started (GUI)
**1. Denoise the data with pretrained networks**
```
python -m src.GUI.test_GUI
```
As the network is trained on different dataset of denoising,
for optimal performance, one might have to train the network.

**2. Train the network**

Will be updated soon.

## Getting Started (code)
**1. Train SUPPORT**
```
python -m src.train --exp_name mytest --noisy_data ./data/sample_data.tif
```
For more options, please refer to the manual through the following code.
```
python -m src.train --help
```

**2. Inferece with SUPPORT**

Edit src/test.py file to change the name of the dataset, and run the following code.
```
python -m src.test
```

## Data availability
Dataset for volumetric structured imaging of *penicillium* and calcium imaging of larval zebrafish can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.7330257).

## Contributors
We are happy to help any questions or requests.
Please contact to following authors to get in touch!
* Minho Eom (djaalsgh159@kaist.ac.kr)
* Seungjae Han (jay0118@kaist.ac.kr)

## Citation
Eom, M. et al. Statistically unbiased prediction enables accurate denoising of voltage imaging data. Preprint at *bioRxiv* (2022).
