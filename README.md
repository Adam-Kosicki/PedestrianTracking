# PedestrianTracking
Video sources: https://utexas.box.com/s/bds4ytr5pm0rw9k7t4y8avzkk39oj6uq

# Environment Setup
In this setup, we used an Ubuntu 20.04 machine with an Nvidia RTX A4000 GPU. However, these instructions are made to work on any distribution of Linux using CUDA-Enabled GPUs (look [here](https://developer.nvidia.com/cuda-gpus) to make sure your GPU is compatible).

## Drivers
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
In addition, to utilize the Nvidia GPU with YOLOv9, we'll need to install PyTorch and CUDA (instructions provided)
https://developer.nvidia.com/cuda-gpus