<!-- <img src='imgs/horse2zebra.gif' align="right" width=384>

<br><br><br>
-->
# Unsupervised Color Enhancement

Tensorflow implementation for learning an image-to-image color enhancement using CycleGAN structure (unsupervised).

For image example:
![color_enhance](/assets/color_enhance.png)

It learns color affine transform function for each pixel in CIE L\*a\*b\*.
Network structure for transformation network looks:
![affine_structure](/assets/affine_structure.png)

This implementation is based on CycleGAN-tensorflow of xhujoy (https://github.com/xhujoy).
This repository contains train and test codes for reproduce.
Pretrained network model and dataset will be distributed soon.

--------------------------

## Prerequisites
- tensorflow r1.0 or higher version
- numpy 1.11.0
- scipy 0.17.0
- pillow 3.3.0

## Getting Started
### Installation
- Install tensorflow from https://github.com/tensorflow/tensorflow
- Clone this repo:
```bash
git clone https://github.com/JunhoJeon/unsupervised-color-enhance
cd CycleGAN-tensorflow
```

## Training and Test Details
To train a model,  
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_dir=/path/to/data/
```
Models are saved to `./checkpoints/` (can be changed by passing `--checkpoint_dir=your_dir`).  

To test the model,
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_dir=/path/to/data/ --phase=test --which_direction=AtoB/BtoA
```

## Reference
- The tensorflow implementation of CyelcGAN (which this repository forked from), https://github.com/xhujoy/CycleGAN-tensorflow
- The torch implementation of CycleGAN, https://github.com/junyanz/CycleGAN
- The tensorflow implementation of pix2pix, https://github.com/yenchenlin/pix2pix-tensorflow
