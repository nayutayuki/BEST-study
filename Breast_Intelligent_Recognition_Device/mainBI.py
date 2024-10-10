import argparse
import os
import time
import numpy as np

import shutil
import sys
import pdb
import logging

import torch
from torch.nn import DataParallel, Linear
from torch.backends import cudnn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from data_utils.dataloader import BIRADSData, CenterBIData, InternalBIData, RetroTestData
import data_utils.transforms as tr
from utils import *
from transformers import get_linear_schedule_with_warmup

###########################################################################
"""
                The main function of AI + BC
                      Python 3
                    pytorch 1.6.0
                   author: Tao He
              Institution: Sichuan University
               email: taohe@stu.scu.edu.cn
"""
###########################################################################
# 1. 裁剪ROI区域+严格裁剪结节+宽松裁剪结节；
# 2. 采用多任务学习（结节属性的分类作为辅助任务）
# 3. 输入大小限定为128*128，为了快速验证模型效果

parser = argparse.ArgumentParser(description='PyTorch Classification for BC')
parser.add_argument('--model_name',  default='EfficientnetB3', type=str )
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--start_epoch', default=1,  type=int)
parser.add_argument('-b', '--batch_size', default=64, type=int)
parser.add_argument('--lr', default=0.0001, type=float )
parser.add_argument('--momentum',  default=0.9, type=float )
parser.add_argument('--resume',  default='',  type=str )
parser.add_argument('--weight_decay', default=0.0005, type=float )
parser.add_argument('--save_dir', default='../SavePath/20230413_bigpaper', type=str)
parser.add_argument('--gpu', default='3', type=str)
parser.add_argument('--patient', default=5, type=int)
parser.add_argument('--untest_epoch', default=0, type=int)
parser.add_argument('--loss_name', default='crossentropy', type=str)
parser.add_argument('--data_path', default='./BC_data', type=str)
parser.add_argument('--classify_name', default='BI7', type=str)   
parser.add_argument('--input_size', default=256, type=int)
parser.add_argument('--plot_step', default=200, type=int)
parser.add_argument('--inchannel', default=4, type=int)
parser.add_argument('--grad_list', default=["fc", "Mixed_7"], type=list) 
parser.add_argument('--aux_epoch', default=0, type=int)  
parser.add_argument('--test_flag', default=-1, type=int)   
parser.add_argument('--data_name', default="Dazhou", type=str)   
parser.add_argument('--over_sample', default="0_0_0_1_8_15_30", type=str)    
DEVICE = torch.device("cuda" if True else "cpu")

def set_parameter_requires_grad(model, feature_extracting, abbrs):
    if feature_extracting:
        idxs = []
        i = 0
        for name, param in model.named_parameters():
            grad_flag = False
            for abbr in abbrs:
                if abbr in name:
                    grad_flag = True
                    break
            if(not grad_flag):
                idxs.append(i)
            i+=1
        for j, param in enumerate(model.parameters()):
            if j in idxs:
                param.requires_grad = False

def main(args):
    cudnn.benchmark = True
    setgpu(args.gpu)

    ################################ experiments ############################################
    if(args.classify_name == "BI2"):
        label_dict = {'0':0, '1':0, '2':0, '3':1, '4':1, '5':1, '6':1}
        n_class = 2
    elif(args.classify_name == "BI5"):
        label_dict = {'0':0, '1':1, '2':2, '3':3, '4':3, '5':4, '6':4}
        n_class = 5
    elif(args.classify_name == "BI7"):
        label_dict = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6}
        n_class = 7

    over_samplestr = args.over_sample.split("_")
    over_sample = [int(i) for i in over_samplestr]
    if len(over_sample)!=n_class:
        over_sample = None
    ##########################################################################################
    if args.model_name == 'ResNet18':
        from models.resnet import resnet18
        net = resnet18(pretrained=True, inchannels=args.inchannel, num_classes=n_class)
    elif args.model_name == 'inception_v3':
        from models.pretrain import InceptionV3
        net = InceptionV3(4, n_class)
    elif args.model_name == "EfficientnetB3":
        from models.pretrain import EfficientnetB3
        net = EfficientnetB3(4, n_class)

    if args.loss_name == 'crossentropy':
        loss = torch.nn.CrossEntropyLoss()
    elif args.loss_name == 'weightedCE':
        weight = torch.tensor([0.7, 0.3])
        loss = torch.nn.CrossEntropyLoss(weight=weight)
    elif args.loss_name == 'L1Loss':
        loss = torch.nn.L1Loss()
    elif args.loss_name == 'FocalLoss':
        from models.losses import FocalLoss
        loss = FocalLoss(class_num=n_class)
    ###########################################################################################

    start_epoch = args.start_epoch
    save_dir = args.save_dir
    logging.info(args)
    if args.resume:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        net.load_state_dict(checkpoint['state_dict'])

    net = net.to(DEVICE)
    loss = loss.to(DEVICE)
    if len(args.gpu.split(',')) > 1 or args.gpu == 'all':
        net = DataParallel(net)

    params_to_update = []
    #print("*****The parameters need to be update! *****")
    for name,param in net.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            #print("\t",name)
    #print("**********************************************")
    # testing dataset
    test_transform = transforms.Compose([
            tr.CenterCropMask(),
            tr.Resize(size=(args.input_size,args.input_size)),
            tr.Normalize(phase="Test"),
            tr.ToTensor(),
    ])
    test_dataset = RetroTestData(transform=test_transform, #phase='Test', 
             data_name=args.data_name,
             label_dict=label_dict,
             parent_dir=args.data_path)
    testloader = DataLoader(test_dataset,
             batch_size=args.batch_size,
             shuffle=False,
             num_workers=8)

    eval_dataset = RetroTestData(transform=test_transform, #phase='Val', 
             data_name=args.data_name,
             label_dict=label_dict,
             parent_dir=args.data_path)
    evalloader = DataLoader(eval_dataset,
             batch_size=args.batch_size,
             shuffle=False,
             num_workers=8)
    
    train_transform = transforms.Compose([
                tr.RandomCropMask(),
                tr.Resize(size=(args.input_size,args.input_size)),
                tr.HorizontalFlip(),
                tr.Normalize(phase="Train"),
                tr.ToTensor(),
            ])

    train_dataset = BIRADSData(transform=train_transform, phase='Train', 
                 parent_dir=args.data_path,
                 data_name=args.data_name,
                 label_dict=label_dict,
                 over_sample=over_sample
                 )
    trainloader = DataLoader(train_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=8)

    optimizer = torch.optim.Adam(params_to_update, lr=args.lr, weight_decay=args.weight_decay)
    #optimizer = torch.optim.SGD(params_to_update, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = GradScaler()
    if args.start_epoch == 1:
        scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps= 4*int(train_dataset.__len__()/args.batch_size), 
        num_training_steps= int(train_dataset.__len__()/args.batch_size*args.epochs)
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps= 2, 
        num_training_steps= int(train_dataset.__len__()/args.batch_size*args.epochs)
        )

    break_flag = 0.
    max_acc = 0.
    min_loss = 1000.
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []

    if args.test_flag == 1:
        _, confuse_m, _, _ = evaluation(testloader, net, loss, 0, "Test", n_class, args.test_flag)
        #np.save(os.path.join(save_dir, 'Test_confuse_matrix.npy'), np.array(confuse_m))
    elif args.test_flag == 0:
        _, confuse_m, _, _ = evaluation(evalloader, net, loss, 0, "Test", n_class, args.test_flag)
        #np.save(os.path.join(save_dir, 'Valid_confuse_matrix.npy'), np.array(confuse_m))
    else:
        for epoch in range(start_epoch, args.epochs + 1):
            cur_max_acc, cur_train_acc, cur_train_loss, cur_val_acc, cur_val_loss = train(trainloader, 
                            evalloader, net, loss, epoch, optimizer, 
                            scaler, args.plot_step, save_dir, 
                            max_acc, n_class,scheduler) 
            train_acc.append(cur_train_acc)
            train_loss.append(cur_train_loss)
            val_acc.append(cur_val_acc)
            val_loss.append(cur_val_loss)
            if cur_max_acc > max_acc:
                max_acc = cur_max_acc
                break_flag = 0
            else:
                break_flag += 1
#             if break_flag > args.patient and epoch > 60:
#                 break
    
    ########### save for plotting ###############
#     np.save(os.path.join("../fig_plot", "train_loss"+"_BI"+str(n_class)+".npy"), np.array(train_loss))
#     np.save(os.path.join("../fig_plot", "train_acc"+"_BI"+str(n_class)+".npy"), np.array(train_acc))
#     np.save(os.path.join("../fig_plot","val_loss"+"_BI"+str(n_class)+".npy"), np.array(val_loss))
#     np.save(os.path.join("../fig_plot","val_acc"+"_BI"+str(n_class)+".npy"), np.array(val_acc))


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, 1)
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train(trainloader, evalloader, net, loss, epoch, optimizer, scaler, plot_step, save_dir, max_acc, n_class, scheduler):
    start_time = time.time()
    net.train()

    train_loss = 0
    step = 0
    total = 0
    correct = 0

    for i, sample in enumerate(trainloader):
        data = sample['image']
        label = sample['label']
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        data, targets_a, targets_b, lam = mixup_data(data, label, 0.5, True)
        optimizer.zero_grad()
        
        with autocast():
            output = net(data)
            cur_loss = mixup_criterion(loss, output, targets_a, targets_b, lam)
        predicted = torch.argmax(output.data, 1)
        total += output.size(0)
        
        correct += (lam * predicted.eq(targets_a).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b).cpu().sum().float())
        train_loss +=cur_loss.item()
        step += 1

        scaler.scale(cur_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
    _, _, cur_train_acc, cur_train_loss = evaluation(trainloader, net, loss, epoch, "Train", n_class, 0)
    logging.info(
    'Train --> Epoch[%d], Step[%d] lr[%.6f], total loss: [%.6f], acc: [%.2f %%], time: %.1f s!'
    % (epoch,i, optimizer.param_groups[0]['lr'], cur_train_loss, 100.*cur_train_acc,time.time() - start_time))
    start_time = time.time()
    cur_acc, confuse_matrix, cur_val_acc, cur_val_loss = evaluation(evalloader, net, loss, epoch, "Test", n_class)
    
    if cur_acc >= max_acc:
        max_acc = cur_acc
        if len(args.gpu.split(',')) > 1 or args.gpu == 'all':
            state_dict = net.module.state_dict()
        else:
            state_dict = net.state_dict()
        torch.save(
            {
                'epoch': epoch,
                'save_dir': save_dir,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
                'args': args
            }, os.path.join(save_dir, 'model.ckpt'))
        np.save(os.path.join(save_dir, str(epoch)+'_confuse_matrix.npy'), np.array(confuse_matrix))
        logging.info(
            '***********************model saved successful************************* !\n'
        )
    return max_acc, cur_train_acc, cur_train_loss, cur_val_acc, cur_val_loss


def evaluation(evalloader, net, loss, epoch, type, n_class, test_flag):
    start_time = time.time()
    net.eval()
    total_loss = []
    preds = []
    targets = []
    correct = 0
    total = 0
    outputs = []
    with torch.no_grad():
        for i, sample in enumerate(evalloader):
            data = sample['image']
            label = sample['label']
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            output = net(data)
            outputs.append(torch.softmax(output,1).cpu().numpy())
            cur_loss = loss(output, label)
            targets.append(label.cpu().numpy())
            preds.append(torch.argmax(output, 1).detach().cpu().numpy())
            total_loss.append(cur_loss.item())
            correct += torch.argmax(output, 1).eq(label).detach().cpu().numpy().sum()
            total += output.size(0)     

    confuse_matrix = get_GradeBI(n_class, np.concatenate(preds, 0), np.concatenate(targets, 0))
    if type == "Test":
        logging.info('BIRADS confuse matrix =======>')
        logging.info(confuse_matrix)
        ### for roc curves
        np.save(os.path.join("../fig_plot", "output_"+str(test_flag)+"_"+str(n_class)+".npy"), np.concatenate(outputs, 0))
        np.save(os.path.join("../fig_plot", "target_"+str(test_flag)+"_"+str(n_class)+".npy"), np.concatenate(targets, 0))

        logging.info(
        'Test --> epoch[%d], total loss: [%.6f], acc: [%.4f], time: %.1f s!'
        % (epoch, np.mean(total_loss), correct/total, time.time() - start_time))
    return correct/total, confuse_matrix, correct/total, np.mean(total_loss)


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_dir = os.path.join(args.save_dir, args.model_name, args.classify_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s,%(lineno)d: %(message)s\n',
                        datefmt='%Y-%m-%d(%a)%H:%M:%S',
                        filename=os.path.join(args.save_dir, 'log.txt'),
                        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    main(args)
    