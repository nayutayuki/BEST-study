'''
@Date: 2019-12-05 03:33:58
@Author: Yong Pi
LastEditors: Yong Pi
LastEditTime: 2021-09-16 08:07:43
@Description: All rights reserved.
'''
import argparse
import random

import numpy as np
import torch

from utils.gpu import set_gpu
from utils.parse import parse_yaml

import os
import time
os.environ["TZ"] = "UTC-8"
time.tzset()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


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
    model = Model(config)
    model.run()
