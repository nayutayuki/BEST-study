# BMM and ICM Training Code

## Project Description
EfficientNetV2 is an optimized version, which has smaller parameters than EfficientNet, better parameter efficiency, and faster training speed. In this study, EfficientNetV2 was used to develop two binary classification models:

- **BMM Model:** Distinguishes benign and malignant lesions.
- **ICM Model:** Distinguishes invasive carcinoma from ductal carcinoma in situ.

The training process for both models is the same. Breast ultrasound images were obtained from 5 hospitals, using a total of 2,792 ultrasound images for BMM (1,043 benign and 1,749 malignant) and 1,749 malignant ultrasound images for ICM (394 ductal carcinoma in situ and 1,355 invasive carcinoma). The data was divided into training, validation, and test sets in a ratio of 7:2:1. Images were preprocessed into 3×299×299, and data augmentation was performed.

The training process was 300 epochs, with a learning rate of 0.001, a weight decay rate of 0.0005, and a small batch size of 8. Dropout was applied to the last fully connected layer of the second subnet, and the drop probability was set to 0.5 to alleviate overfitting.

## Installation Instructions
Ensure you have the following dependencies installed:
- Pytorch 1.7
- GPU
- CUDA 11.0

## Usage Instructions
### Data and Model Configuration
Configurations are located in the `cfgs` directory, e.g., `invasive.yaml`:
```yaml
task: breast usg malignant classification

data:
  data_dir: /.../Data
  train_file: /.../invasive/train.txt
  val_file: /.../invasive/val.txt
  test_file: /.../invasive/test.txt
  external_file: /.../invasive/external.txt
  color_channels: 3
  mode: gray
  num_classes: 2

train:
  batch_size: 32
  num_workers: 8
  pin_memory: true
  aug_trans:
    trans_seq: [fixed_resize, random_horizontal_flip, to_tensor]
    flip_prob: 0.5
    fixed_resize:
      size: [299, 299]

eval:
  batch_size: 32
  num_workers: 8
  pin_memory: true
  ckpt_path: None
  aug_trans:
    trans_seq: [fixed_resize, to_tensor]
    fixed_resize:
      size: [299, 299]

logging:
  use_logging: true
  use_tensorboard: true

optim:
  num_epochs: 200
  optim_method: adam
  sgd:
    base_lr: 1e-3
    momentum: 0.9
    weight_decay: 5e-4
    nesterov: false
  adam:
    base_lr: 1e-2
    betas: [0.9, 0.999]
    weight_decay: 1e-4
    momentum: 0.9
    amsgrad: false
  use_lr_decay: false
  lr_decay_method: lambda
  cosine: None
  warmup_cosine: None

criterion:
  criterion_method: cross_entropy

network:
  backbone: resnet50
  model_suffix: invasive
  drop_prob: 0.5
  use_parallel: false
  seed: 22
  num_gpus: 0
```
## Training Command
```
SEED=42
GPU=0
python main.py --config cfgs/<config file name>.yaml --mode cdfi --backbone <backbone_name> --gpu $GPU --seed $SEED --model sil_model
python main.py --config cfgs/<config file name>.yaml --mode gray --backbone <backbone_name> --gpu $GPU --seed $SEED --model sil_model
```

## Contributing

Please create a pull request if you would like to contribute to the project. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.

## Contact

For questions or inquiries, please contact XXXXXX.
