## Step-by-step instructions for beginners, who do not know Python, Anaconda, etc.

1. Download and install the Anaconda program (https://conda.io/projects/conda/en/latest/user-guide/install/windows.html). You can find the "Anaconda Powershell Prompt" program after installation.
<p align="center">
  <img src="https://github.com/NICALab/SUPPORT/assets/31270778/7cfb1e22-7d86-4500-9e69-4a3f86facca7" width="700">
  <img src="https://github.com/NICALab/SUPPORT/assets/31270778/4e842f32-b0ea-4450-b530-749ca0c70d97" width="500">
  <br>Anaconda Powershell Prompt
</p>
2. Execute it and direct it to the folder where you want to install our program. (use "cd" and "ls" commands to change the current location)
3. Clone the repository by entering the following command: "git clone https://github.com/NICALab/SUPPORT.git"
<p align="center">
  <img src="https://github.com/NICALab/SUPPORT/assets/31270778/d0efb21d-1812-450a-848a-6e2ba7dbee35" width="700">
</p>
4. Enter the following command: "cd ./SUPPORT"
5. Enter the following command: "conda env create -f env.yml"
<p align="center">
  <img src="https://github.com/NICALab/SUPPORT/assets/31270778/04ce2f8a-fe9a-40df-9cf5-68fe48ebc9a3" width="700">
</p>
6. Enter the following command: "conda activate SUPPORT"
7. Enter the following command: "conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia"
<p align="center">
  <img src="https://github.com/NICALab/SUPPORT/assets/31270778/e7b5a656-29f5-41d8-a9e5-c4046da03594" width="700">
</p>

## After that, let's first run our default model to briefly check if it works!
To run graphic-based program, enter the following command: "python -m src.GUI.train_GUI".
<p align="center">
  <img src="https://github.com/NICALab/SUPPORT/assets/31270778/6286fc13-e2d1-47b7-901a-8a5696d4f56d" width="700">
  <img src="https://github.com/NICALab/SUPPORT/assets/31270778/0299b116-4f61-4fe4-9347-ea524b3b7e32" width="700">
</p>

You can load your noisy image, and then click **Run**!

## Trouble shooting

If you meet the following error,
```
Torch not compiled with cuda enabled
```
Please follow the steps below.

1. pip uninstall torch
2. pip cache purge
3. pip install torch -f https://download.pytorch.org/whl/torch_stable.html


## Notes
1. Default model does not mean the model we are proud of. It is just for (1) Roughly see how the denoised outcome looks like, (2) Check if there is any problem in program running. The best is, training the SUPPORT model on your own data, and then test it to the same data. Current default model is trained with different data to yours (of course), the performance will not that good.
2. Since we basically developed in the Linux environment, there may be some issues in Windows. Please let us know if you face any problems!!! Email is jay0118@kaist.ac.kr & djaalsgh159@kaist.ac.kr.
