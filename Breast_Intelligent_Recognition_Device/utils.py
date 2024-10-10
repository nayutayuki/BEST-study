import os
import torch
from torch import nn
import pdb
import numpy as np
from PIL import Image
import cv2
import logging

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_metrics(gt, pred, average="binary"):
    acc = accuracy_score(gt, pred)
    pre = precision_score(gt, pred, average=average)
    rec = recall_score(gt, pred, average=average)
    f1 = f1_score(gt, pred, average=average)
    
    return pre, rec, f1, acc


def get_GradeBI(n_class, preds, targets):
    # BIRADS 分类
    formats = n_class
    results = np.zeros((formats, formats)) 
    for pred, target in zip(preds, targets):
        results[target, pred] += 1

    for i in range(formats):
        results[i] = np.round(results[i] / (sum(results[i])) , 4)
    return results


def setgpu(gpus):
    if gpus == 'all':
        gpus = '0,1,2,3'
    print('using gpu ' + gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    return len(gpus.split(','))


def get_Metric(preds, targets):
    # 良恶性分类
    bresults = np.zeros((2, 2)) 
    for pred, target in zip(preds, targets):
        if pred > 1:
            pred = 1
        else:
            pred = 0
        if target > 1:
            target = 1
        else:
            target = 0
        
        bresults[target, pred] += 1
    for i in range(2):
        bresults[i] = np.round(bresults[i] / (sum(bresults[i])) , 4)

    # BIRADS 分类
    formats = max(targets) + 1
    results = np.zeros((formats, formats)) 
    for pred, target in zip(preds, targets):
        results[target, pred] += 1
    
    T = 0
    ALL = 0
    for i in range(formats):
        results[i] = np.round(results[i] / (sum(results[i])) , 4)
        T += results[i,i]
        ALL += sum(results[i])
    if formats <=2:
        errors = 0
    else:
        errors = sum(results[0,2:])+sum(results[1,2:])+sum(results[2,:2])+sum(results[3,:2])+sum(results[4,:2])
    return bresults, results, errors



def save_fig(fake, file_path):
#     plt.imshow(fake.detach().cpu().numpy()[0].transpose([1,2,0]))
    # plt.imsave(file_path, fake.detach().cpu().numpy()[0].transpose([1,2,0]))

    fake = fake.detach().cpu().numpy()[0].transpose([1,2,0])
    fake_t = np.copy(fake)
    fake_t[:,:,0] = fake[:,:,2]
    fake_t[:,:,2] = fake[:,:,0]

    cv2.imwrite(file_path, fake_t)
    
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def get_logger(log_path):
    parent_path = os.path.dirname(log_path)  # get parent path
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(level=logging.INFO,
                    filename=log_path,
                    format='%(levelname)s:%(name)s:%(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    logger = logging.getLogger()
    logger.addHandler(console)
    return logger