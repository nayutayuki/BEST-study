molecular subtype model (MSM)

Description:
   To solve the “degradation” problem of deep neural networks, ResNet invented “fast connection”, which not only adjusts the model structure by designing the residual block structure, but also greatly eliminates the difficulty of training neural networks with excessive depth. In this project, we used ResNet18 to develop MSM to distinguish luminal A, luminal B (HER2 negative), luminal B (HER2 positive), human epidermal growth factor receptor 2 (HER2) enriched and triple negative. The data were from 4 hospitals, with a total of 9,207 breast ultrasound images, including 1,141 luminal A images, 2,095 luminal B (HER2 negative) images, 1,179 luminal B (HER2 positive) images, 912 HER2 enriched images and 1,029 triple negative images. The data was divided into training set, validation set and test set according to the ratio of 8:1:1. The size of ultrasonic image was reduced to the unified size 3×128×128. Also using ImageNet as the pre-training set, Adam was chosen as the final optimizer. The model was trained with a learning rate is 0.001, batch size of 256 and total epochs of 100.

Installation:
  conda virtual environment is recommended.
  run:
    conda create -n MSM python=3.9
    conda activate MSM
    pip install opencv-python
    pip install torch
    pip install torchvision
    pip install timm
    pip install loguru
    pip install numpy
    pip install pandas

Code Running
  run:
    python code.py