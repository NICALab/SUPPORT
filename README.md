# SUPPORT: statistically unbiased prediction enables accurate denoising of voltage imaging data

<div align="center"><img src="thumbnail.png" width="80%"/></div>

## About SUPPORT
Here we present SUPPORT (**S**tatistically **U**nbiased **P**rediction utilizing s**P**ati**O**tempo**R**al information in imaging da**T**a), a self-supervised denoising method for voltage imaging data. FRECTAL is based on the insight that a pixel value in voltage imaging data is highly dependent on its spatially neighboring pixels in the same time frame even when its temporally adjacent frames do not provide useful information for statistical prediction. Such spatiotemporal dependency is captured and utilized to accurately denoise voltage imaging data in which the existence of the action potential in a time frame cannot be inferred by the information in other frames. We show, through simulation and experiments, that FRECTAL enables precise denoising of voltage imaging data while preserving the underlying dynamics in the scene. 

## Getting Started (code)
```
python -m src.train --exp_name mytest --noisy_data mydata.tif
```

## Getting Started (GUI)

## Citation
TODO
