# INSTALLATION GUIDE
This is the Installation guide for the overall repository.

## Install NVIDIA-driver
GIGABYTE based BIOS setting
- First, DO NOT PLUG IN your GPU until the driver is set up.
- Internal Graphic : Auto > Enable
- Display Priority : PCIe > Internal

NVIDIA driver download .run file : [click here](http://www.nvidia.co.kr/Download/index.aspx)

If you click download from the above site, you will get a .run file format for installing drivers.

### 1. Stop display manager

Before you run the .run file, you first need to stop your Xserver display manager.

Press [Ctrl] + [Alt] + [F1], enter the script below

```bash
$ service --status-all | grep dm

(Result) [+] [:dm]
```

The part described as [:dm] is your display manager.

Substitute the [:dm] part below with the result of the script above.

```bash
$ sudo service [:dm] stop

(Result) * Stopping Light Display Manager [:dm]
```

### 2. Run the nvidia-driver installer

Run the code below. Press 'Yes' for every option they ask.

```bash
$ sh <DIR where you downloaded the .run file>/NVIDIA-Linux_x86_64-375.20.run
```

After you have successfully installed, you shall see the same results when typing the code below.

```bash
$ nvidia-smi
```

```bash
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 375.20                 Driver Version: 375.20                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  TITAN X (Pascal)    Off  | 0000:4B:00.0     Off |                  N/A |
| 67%   86C    P2   249W / 250W |   5026MiB / 12221MiB |     82%      Default |
+-------------------------------+----------------------+----------------------+
|   1  TITAN X (Pascal)    Off  | 0000:4C:00.0      On |                  N/A |
| 88%   90C    P2   225W / 250W |   3842MiB / 12213MiB |     78%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

```

### 3. Reboot the system

```bash
$ sudo reboot
```

## Install CUDA toolkit

You can skip the step above and automatically install the driver within CUDA installation.

Check the details below.

### * Download the CUDA download .run file from the CUDA download page

CUDA download page : [click here](https://developer.nvidia.com/cuda-downloads)

Before executing the file, stop the display manager by following the description above.

```bash
$ sudo sh <DIR where you downloaded the .run file>/cuda_8.0.44_linux.run
```

### * Link your CUDA in .bashrc

```bash
$ sudo apt-get install vim

$ git clone https://github.com/amix/vimrc.git ~/.vim_runtime

# Awesome version
$ sh ~/.vim_runtime/install_awesome_vimrc.sh

# Basic version
$ sh ~/.vim_runtime/install_basic_vimrc.sh
```

Open your ~/.bashrc file.

```bash
vi ~/.bashrc

# Press Shift + G, Add the lines on the bottom

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_HOME=/usr/local/cuda
```

To check if the CUDA toolkit is successfully installed, type the line below.

```bash
$ nvcc --version

* (Result)
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Sun_Sep__4_22:14:01_CDT_2016
Cuda compilation tools, release 8.0, V8.0.44
```

## Install cuDNN library kit

cuDNN download page : [click here](https://developer.nvidia.com/rdp/cudnn-download)

(Membership is required, just sign in!)

Download the newest cuDNN v5.1.

```bash
$ cd <DOWNLOAD DIR>
$ tar -zxvf ./cudnn-8.0-linux-x64-v5.1.tgz
$ sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
$ sudo chmod a+r /usr/local/cuda/include/cudnn.h
$ sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```

## Install Tensorflow

Tensorflow install page : [click here](https://www.tensorflow.org/get_started/os_setup)

```bash
$ sudo apt-get install python-pip python-dev
$ pip install --upgrade pip 
$ pip install tensorflow-gpu
```

## Install Torch

Torch install page : [click here](http://torch.ch/docs/getting-started.html)

```bash
$ git clone https://github.com/torch/distro.git ~/torch --recursive
$ cd ~/torch; bash install-deps;
$ ./install.sh
$ source ~/.bashrc
```

## Now, Enjoy!
