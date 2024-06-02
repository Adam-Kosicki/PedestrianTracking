# PedestrianTracking
Video sources: https://utexas.box.com/s/bds4ytr5pm0rw9k7t4y8avzkk39oj6uq

# Environment Setup
In this setup, we used an Ubuntu 20.04 machine with an Nvidia RTX A4000 GPU. However, these instructions are made to work on any distribution of Linux using CUDA-Enabled GPUs (look [here](https://developer.nvidia.com/cuda-gpus) to make sure your GPU is compatible). The steps below will guide you to setup the environment necessary to run PyTorch-based tools using CUDA.

## 1.) Drivers
Ensure the NVidia drivers are up to date (sources found [here](https://ubuntu.com/server/docs/nvidia-drivers-installation)).

First, we will verify that our system has the necessary drivers.
```
cat /proc/driver/nvidia/version
```

If the command returns returns the name of the GPU (e.g., _"GPU 0: NVIDIA ..."_), then you may skip to the [Pytorch/CUDA setup](#pytorch/cuda)! However, note that eventhough the drivers may be present, it doesn't mean they are up-to-date. If you are unsure, it's recommended to follow the steps below.

Remove nvidia orphan dependencies
```
sudo apt autoremove nvidia* --purge
```

Run nvidia's package uninstaller, if any, to ensure full removal
```
sudo /usr/bin/nvidia-uninstall
```

Run CUDA's package uninstaller, if any, to prevent mismatching versions
```
sudo /usr/local/cuda-X.Y/bin/cuda-uninstall
```

Do a full update of Ubuntu (reboot, if necessary)
```
sudo apt update
```
```
sudo apt upgrade
```

View nvidia drivers available for the GPU we have. The command should retun a list of drivers compatible with your GPU (model will be shown as well). However, only a few of them will be marked as _"recommended"_ (e.g., _"driver: nvidia-driver-XXX - distro non-free recommended"_).
```
sudo apt install alsa-utils ubuntu-drivers-common
```
```
ubuntu-drivers devices
```

Finally, we will automatically install this recommended driver and reboot the system to ensure the settings are applied.
```
sudo ubuntu-drivers autoinstall
```
```
sudo reboot
```

## Pytorch/CUDA
For version containerization purposes, we will be utilizing a conda environment (learn how to do that [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)).

Once conda is ready for use, create a conda environment with the desired version of python (named 'peds_tracker' in this case) and activate it.
```
conda create -n peds_tracker python=3.11
```
```
conda activate peds_tracker
```

Install the cuDNN, which is a GPU-accelerateed library for deep neural networks.
```
conda install -c nvidia cuda-nvcc
```
```
conda install -c "nvidia/label/cuda-11.3.0" cuda-nvcc
```

Install the CUDA toolkit, which is needed by PyTorch for GPU acceleration.
```
conda install -c anaconda cudatoolkit
```

Finally, install the latest version of PyTorch with GPU support.
```
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
```

That's it! You can test whether the setup was successful with the following code:
```
import torch

print("Check CUDA availability:", torch.cuda.is_available())
print("Device used by CUDA:", torch.cuda.get_device_name())

!nvcc --version

device = None
if(torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("using", device, "device")
```

If everything is setup correctly, you should get something like this:
```
Check CUDA availability: True
Device used by CUDA: NVIDIA RTX A4000

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0

using cuda device
```
