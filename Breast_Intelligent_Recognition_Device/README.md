## Project Introduction
This project aims to classify image data using deep learning techniques, primarily applied in the medical imaging field, especially in breast cancer imaging (BIRADS classification). The project adopts a multi-task learning model, incorporating various network architectures (such as EfficientNet, ResNet, etc.) and is implemented based on the PyTorch framework.

## Features
Multi-task Learning: Combines image cropping and classification tasks to achieve feature extraction and classification.  
Multiple Classification Models: Supports various model architectures, including ResNet18, Inception_v3, EfficientNetB3, and more.  
Model Training and Evaluation: Provides training and evaluation scripts that allow parameter settings according to user needs.  

## System Requirements
### Hardware requirements
This package requires only a standard computer with enough RAM to support the in-memory operations.
### Software requirements
#### OS Requirements
- Linux: Ubuntu 22.04
#### Python Dependencies
albumentations==1.4.18
chardet==4.0.0
ml_collections==0.1.1
numpy==2.1.2
Pillow==10.4.0
scikit_learn==1.5.2
scipy==1.14.1
skimage==0.0
torch==2.3.0
torchvision==0.18.0
transformers==4.45.2

## Installation Guide
Clone the project to your local machine:  
git clone https://github.comnayutayuki/breast_cancer_classification.git  
cd breast_cancer_classification  
pip install -r requirements.txt   
Installation will consume several minutes.

## Usage Instructions
Place the dataset in the specified path, for example: '/root/workspace/data/BC_data/developmentSet'.  
Run the training script:  
python mainBI.py --model_name EfficientnetB3 --epochs 100 --batch_size 64 --data_path "./BC_data"  

### Optional parameters:
--model_name: The name of the model to use, such as ResNet18, EfficientnetB3, etc.  
--epochs: Sets the number of training epochs, with a default value of 100.  
--batch_size: Sets the batch size, with a default value of 64.  
Other parameters can be found in the mainBI.py file.  

### File Descriptions
mainBI.py: The main training and evaluation script, which includes model definitions, data loading, training, and testing processes.  
utils.py: Contains various utility functions, such as GPU settings, evaluation metric calculations, and image saving.  
data_utils: The data loading and preprocessing module, which includes functionalities like data augmentation.  
retro_result.npy and retro_result.txt: Files used to store training or testing results.  

## Demo
Run: python mainBI.py --model_name EfficientnetB3 --epochs 100 --batch_size 64 --data_path "./BC_data"  
You will get the result file in txt format in the root directory of this project in the following form:
Image Name----real label----predict label