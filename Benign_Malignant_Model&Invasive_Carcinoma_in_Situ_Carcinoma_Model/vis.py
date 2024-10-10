'''
@Date: 2019-12-05 03:33:58
@Author: Yong Pi
LastEditors: Yong Pi
LastEditTime: 2021-09-14 08:42:47
@Description: All rights reserved.
'''
import argparse
import random

import numpy as np
import torch

from utils.gpu import set_gpu
from utils.parse import parse_yaml
import cv2
import os
import time
os.environ["TZ"] = "UTC-8"
time.tzset()
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def save_vis(data_loader, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    count = 0
    for input_tensor, target in data_loader:
        input_tensor, target = input_tensor.to(inference.device), target.to(inference.device)
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda)
        targets = None
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        for i in range(grayscale_cam.shape[0]):
            count+=1
            grayscale = grayscale_cam[i, :]
            rgb_img = input_tensor.cpu().numpy()[i, :].transpose([1,2,0])
            if rgb_img.shape[2]==4:
                rgb_img = rgb_img[:,:,:3]
            visualization = show_cam_on_image(rgb_img, grayscale, use_rgb=True)
            cv2.imwrite(os.path.join(save_path, "%d.jpg"%count), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Classification')
    parser.add_argument('--seed', type=int, default=22,
                        help='random seed for training. default=22')
    parser.add_argument('--use_cuda', default='true', type=str,
                        help='whether use cuda. default: true')
    parser.add_argument('--use_parallel', default='false', type=str,
                        help='whether use cuda. default: false')
    parser.add_argument('--gpu', default='all', type=str,
                        help='use gpu device. default: all')
    parser.add_argument('--config', default='cfgs/default.yaml', type=str,
                        help='configuration file. default=cfgs/default.yaml')
    parser.add_argument('--model', default='sil_model', type=str,
                        help='choose model. default=sil_model')
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help='choose net. default=resnet50')
    parser.add_argument('--mode', default='gray', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--lr', default='1e-2', type=float)
    args = parser.parse_args()

    num_gpus = set_gpu(args.gpu)
    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    config = parse_yaml(args.config)
    config['data']["mode"] = args.mode
    config['optim']["adam"]["base_lr"] = args.lr
    
    network_params = config['network']
    network_params['seed'] = args.seed
    network_params['device'] = "cuda" if str2bool(args.use_cuda) else "cpu"
    network_params['use_parallel'] = str2bool(args.use_parallel)
    network_params['num_gpus'] = num_gpus
    network_params['backbone'] = args.backbone
    if args.model == "sil_model":
        from models.sil_model import Model
    else:
        raise NotImplementedError
    config['eval']['ckpt_path'] = args.ckpt
    inference = Model(config)
    # import pdb;pdb.set_trace()
    model = inference.net
    target_layers = [model.features[7][-2]]
    save_dir = "vis"
    
    save_path = os.path.join(save_dir, "%s_%s_%s_val"%(args.backbone, args.mode, args.config[5:-5]))
    save_vis(inference.val_loader, save_path)
    # save_path = os.path.join(save_dir, "%s_%s_%s_val"%(args.backbone, args.mode, args.config[5:-5]))
    # save_vis(inference.val_loader, save_path)
    # save_path = os.path.join(save_dir, "%s_%s_%s_val"%(args.backbone, args.mode, args.config[5:-5]))
    # save_vis(inference.val_loader, save_path)
    
