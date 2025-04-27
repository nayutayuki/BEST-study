## Project Introduction

In this project, we used ResNet18 to develop MSM to distinguish luminal A, luminal B (HER2 negative), luminal B (HER2 positive), human epidermal growth factor receptor 2 (HER2) enriched and triple negative. The data were from 4 hospitals, with a total of 9,207 breast ultrasound images, including 1,141 luminal A images, 2,095 luminal B (HER2 negative) images, 1,179 luminal B (HER2 positive) images, 912 HER2 enriched images and 1,029 triple negative images. 

## System Requirements

### Hardware requirements
This package requires only a standard computer with enough RAM to support the in-memory operations.

### Software requirements

#### OS Requirements
    Linux: Ubuntu 22.04

#### Python Dependencies
    python==3.8.13
    numpy==1.24.4
    pandas==1.4.3
    torch==1.13.0
    torchvision==0.14.0
    opencv-python==4.5.5.64
    timm==1.0.8
    loguru==0.7.2


## Installation Guide
Clone the project to your local machine.

conda virtual environment is recommended.

    conda create -n MSM python=3.8.13
    conda activate MSM
    pip install opencv-python
    pip install torch
    pip install torchvision
    pip install timm
    pip install loguru
    pip install numpy
    pip install pandas

Installation will consume several minutes.

## Usage Instructions

Run the following script:

    python MSM.py

Using ImageNet as the pre-training set, Adam was chosen as the final optimizer. 

The model was trained with a learning rate is 0.001, batch size of 256 and total epochs of 100.