'''
@Date: 2019-12-05 03:32:00
@Author: Yong Pi
LastEditors: Yong Pi
LastEditTime: 2021-07-14 05:31:20
@Description: All rights reserved.
'''
import os
import time
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from utils import metrics
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
from datasets.usg import Invasive
from torch.utils.data import DataLoader
import datasets.transforms as extend_transforms


class Model(object):

    def __init__(self, config):
        self.config = config
        # create dataset
        self._create_dataset()
        # create net
        self._create_net()
        # logger and writer
        self._create_log()
        # create optimizer
        self._create_optimizer()
        # create criterion
        self._create_criterion()
        if config['eval']['ckpt_path']!="None":
            self.load(config['eval']['ckpt_path'])

    def _create_net(self):
        network_params = self.config["network"]
        # loading network parameters
        self.device = torch.device(network_params['device'])
        self.epochs = self.config['optim']['num_epochs']
        backbone = self.config['network']['backbone']
        color_channels = self.config["data"]["color_channels"]
        
        if backbone=="resnet34":
            self.net = models.resnet34(weights="DEFAULT")
            self.net.fc = nn.Linear(512, self.config["data"]["num_classes"])
            if color_channels!=3:
                self.net.conv1 = nn.Conv2d(color_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif backbone=="resnet50":
            self.net = models.resnet50(weights="DEFAULT")
            self.net.fc = nn.Linear(2048, self.config["data"]["num_classes"])
            if color_channels!=3:
                self.net.conv1 = nn.Conv2d(color_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif backbone=="resnet101":
            self.net = models.resnet101(weights="DEFAULT")
            self.net.fc = nn.Linear(2048, self.config["data"]["num_classes"])
            if color_channels!=3:
                self.net.conv1 = nn.Conv2d(color_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif backbone=="inception_v3":
            self.net = models.inception_v3(weights="DEFAULT", transform_input=False)
            self.net.fc = nn.Linear(2048, self.config["data"]["num_classes"])
            if color_channels!=3:
                self.net.Conv2d_1a_3x3.conv = nn.Conv2d(color_channels, 32, kernel_size=3, stride=2, bias=False)
        elif backbone=="efficientnet_v2_s":
            self.net = models.efficientnet_v2_s(weights="DEFAULT", transform_input=False)
            self.net.classifier[1] = nn.Linear(1280, self.config["data"]["num_classes"])
            # import pdb;pdb.set_trace()
            if color_channels!=3:
                self.net.features[0][0] = nn.Conv2d(color_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
        elif backbone=="convnext_small":
            self.net = models.convnext_small(weights="DEFAULT", transform_input=False)
            self.net.classifier[2] = nn.Linear(768, self.config["data"]["num_classes"])
            if color_channels!=3:
                self.net.features[0][0] = nn.Conv2d(color_channels, 96, kernel_size=4, stride=4, bias=False)
        elif backbone=="swin_t":
            self.net = models.swin_t(weights="DEFAULT")
            self.net.head = nn.Linear(768, self.config["data"]["num_classes"], bias=True)
            if color_channels!=3:
                self.net.features[0][0] = nn.Conv2d(color_channels, 96, kernel_size=4, stride=4, bias=False)
        else:
            raise NotImplementedError("backbone %s not implemented"%backbone)
        
        if network_params['use_parallel']:
            self.net = nn.DataParallel(self.net)
        else:
            self.net = self.net.to(self.device)

    def _create_log(self):
        config = self.config
        model_name = config['network']['backbone']
        model_suffix = config['network']['model_suffix']+"_"+config['data']['mode']
        seed = "_"+str(config['network']['seed'])

        # loading logging parameters
        logging_params = config['logging']
        timestamp = time.strftime("_%Y-%m-%d_%H-%M-%S", time.localtime())
        self.ckpt_path = os.path.join('results', model_name, model_suffix+seed+timestamp, 'ckpt')
        self.tb_path = os.path.join('results', model_name, model_suffix+seed+timestamp, 'tensorboard')
        self.log_path = os.path.join('results', model_name, model_suffix+seed+timestamp, 'log')

        if logging_params['use_logging']:
            from utils.log import get_logger
            from utils.parse import format_config

            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path)
            self.logger = get_logger(self.log_path)
            # self.logger.info(">>>The net is:")
            # self.logger.info(self.net)
            self.logger.info(">>>The config is:")
            self.logger.info(format_config(config))
        if logging_params['use_tensorboard']:
            from tensorboardX import SummaryWriter
            if not os.path.exists(self.tb_path):
                os.makedirs(self.tb_path)
                self.writer = SummaryWriter(self.tb_path)

    def _create_dataset(self):
        def _init_fn(worker_id):
            """Workers init func for setting random seed."""
            np.random.seed(self.config['network']['seed'])
            random.seed(self.config['network']['seed'])

        data_params = self.config['data']
        # making train dataset and dataloader
        train_params = self.config['train']
        train_trans_seq = self._resolve_transforms(train_params['aug_trans'])
        train_dataset = Invasive(root_dir=data_params['data_dir'],
                                txt_file=data_params["train_file"],
                                mode = data_params["mode"],
                                transforms=train_trans_seq)
        self.train_loader = DataLoader(train_dataset,
                                batch_size=train_params['batch_size'],
                                shuffle=True,
                                num_workers=train_params['num_workers'],
                                drop_last=True,
                                pin_memory=train_params['pin_memory'],
                                worker_init_fn=_init_fn)

        # making val dataset and dataloader
        val_params = self.config['eval']
        val_trans_seq = self._resolve_transforms(val_params['aug_trans'])
        val_dataset = Invasive(root_dir=data_params['data_dir'],
                                txt_file=data_params["val_file"],
                                mode = data_params["mode"],
                                transforms=val_trans_seq)
        self.val_loader = DataLoader(val_dataset,
                                batch_size=val_params['batch_size'],
                                shuffle=False,
                                num_workers=val_params['num_workers'],
                                pin_memory=val_params['pin_memory'],
                                worker_init_fn=_init_fn)

        # making test dataset and dataloader
        eval_params = self.config['eval']
        eval_trans_seq = self._resolve_transforms(eval_params['aug_trans'])
        eval_dataset = Invasive(root_dir=data_params['data_dir'],
                                txt_file=data_params["test_file"],
                                mode = data_params["mode"],
                                transforms=eval_trans_seq)
        self.test_loader = DataLoader(eval_dataset,
                                batch_size=eval_params['batch_size'],
                                shuffle=False,
                                num_workers=eval_params['num_workers'],
                                pin_memory=eval_params['pin_memory'],
                                worker_init_fn=_init_fn)

        # making external dataset and dataloader
        eval_params = self.config['eval']
        eval_trans_seq = self._resolve_transforms(eval_params['aug_trans'])
        eval_dataset = Invasive(root_dir=data_params['data_dir'],
                                txt_file=data_params["external_file"],
                                mode = data_params["mode"],
                                transforms=eval_trans_seq)
        self.external_loader = DataLoader(eval_dataset,
                                batch_size=eval_params['batch_size'],
                                shuffle=False,
                                num_workers=eval_params['num_workers'],
                                pin_memory=eval_params['pin_memory'],
                                worker_init_fn=_init_fn)

    def _create_optimizer(self):
        optim_params = self.config['optim']
        if optim_params['optim_method'] == 'sgd':
            sgd_params = optim_params['sgd']
            optimizer = optim.SGD(self.net.parameters(),
                                  lr = sgd_params["base_lr"],
                                  momentum=sgd_params['momentum'],
                                  weight_decay=sgd_params['weight_decay'],
                                  nesterov=sgd_params['nesterov'])
        elif optim_params['optim_method'] == 'adam':
            adam_params = optim_params['adam']
            optimizer = optim.Adam(self.net.parameters(),
                                   lr=adam_params['base_lr'],
                                   betas=adam_params['betas'],
                                   weight_decay=adam_params['weight_decay'],
                                   amsgrad=adam_params['amsgrad'])
        elif optim_params['optim_method'] == 'adadelta':
            adadelta_params = optim_params['adadelta']
            optimizer = optim.Adadelta(self.net.parameters(), lr=adadelta_params['base_lr'], weight_decay=adadelta_params['weight_decay'],)

        # choosing whether to use lr_decay and related parameters
        if optim_params['use_lr_decay']:
            if optim_params['lr_decay_method'] == 'cosine':
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, eta_min=0, T_max=self.config['optim']['num_epochs'])
            if optim_params['lr_decay_method'] == 'lambda':
                lr_lambda = lambda epoch: (1 - float(epoch) / self.config['optim']['num_epochs'])**0.9
                lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
    
    def _create_criterion(self):
        self.best_result = {
            "train": {"epoch": 0, "acc": 0},
            "val": {"epoch": 0, "acc": 0},
            "test": {"epoch": 0, "acc": 0},
            "external": {"epoch": 0, "acc": 0},
                    }
        # choosing criterion
        criterion_params = self.config['criterion']
        if criterion_params['criterion_method'] == 'cross_entropy':
            criterion = nn.CrossEntropyLoss().to(self.device)
        else:
            raise NotImplementedError
        self.criterion = criterion

    def run(self):
        for epoch_id in range(self.config['optim']['num_epochs']):
            self._train(epoch_id)
            if self.config['optim']['use_lr_decay']:
                self.lr_scheduler.step()
            # self._eval(epoch_id,"train")
            self._eval(epoch_id,"val")
            self._eval(epoch_id,"test")
            self._eval(epoch_id,"external")

    def _train(self, epoch_id):
        self.net.train()
        with tqdm(total=len(self.train_loader)) as pbar:
            for data, target in self.train_loader:
                # import pdb;pdb.set_trace()
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.net(data)
                if isinstance(output, models.InceptionOutputs):
                    output = output.logits
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                pbar.update(1)

    def _forward(self, data_loader):
        net = self.net
        net.eval()
        total_loss = 0
        total_output = []
        total_target = []
        num_steps = data_loader.__len__()
        with torch.no_grad():
            with tqdm(total=num_steps) as pbar:
                for data, target in data_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = net(data)
                    if isinstance(output, models.InceptionOutputs):
                        output = output.logits
                    loss = self.criterion(output, target)
                    output = F.softmax(output, dim=1)
                    # convert a 0-dim tensor to a Python number
                    total_loss += loss.item()
                    total_output.extend(output.data.cpu().numpy())
                    total_target.extend(target.data.cpu().numpy())
                    pbar.update(1)
        return total_output, total_target, total_loss


    def _eval(self, epoch, mode="test"):
        logger = self.logger
        if mode == "train":
            data_loader = self.train_loader
        elif mode == "val":
            data_loader = self.val_loader
        elif mode == "test":
            data_loader = self.test_loader
        elif mode == "external":
            data_loader = self.external_loader
        output, target, loss = self._forward(data_loader)
        confusion_matrix = metrics.get_confusion_matrix(output, target)
        num_correct = sum(np.argmax(output, 1) == target)
        acc = num_correct / len(target)
        loss = loss / len(target)
        logger.info("[{}] Epoch:{},confusion matrix:\n{}".format(
            mode, epoch, confusion_matrix))
        logger.info("[{0}] Epoch:{1},{0} acc:{2}/{3}={4:.5},{0} loss:{5:.5}".format(
            mode, epoch, num_correct, len(target), acc, loss))
        self.writer.add_scalar('%s_loss' % mode, loss, epoch)
        self.writer.add_scalar('%s_acc' % mode, acc, epoch)
        results = self.best_result[mode]
        if acc > results["acc"]:
            results["acc"] = acc
            results["epoch"] = epoch
            logger.info('[Info] Epochs:%d, %s accuracy improve to %g' %
                        (epoch, mode, acc))
            if mode=="val":
                self.net.eval()
                snapshot_name = '%s_epoch_%d_loss_%.5f_acc_%.5f_lr_%.10f' % (
                    mode, epoch, loss, acc, self.optimizer.param_groups[0]['lr'])
                torch.save(self.net.state_dict(), os.path.join(
                    self.ckpt_path, snapshot_name + '.pth'))
                # torch.save(self.optimizer.state_dict(), os.path.join(
                #     self.ckpt_path, 'opt_' + snapshot_name + '.pth'))
        else:
            logger.info("[Info] Epochs:%d, %s accuracy didn't improve,\
current best acc is %g, epoch:%g" % (epoch, mode, results["acc"], results["epoch"]))
        return output, target

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        if self.config['network']['use_parallel']:
            self.net.module.load_state_dict(ckpt)
        else:
            self.net.load_state_dict(ckpt)
        print(">>> Loading model successfully from {}.".format(ckpt_path))

    def save(self, epoch):
        if self.config['network']['use_parallel']:
            state_dict = self.net.module.state_dict()
        else:
            state_dict = self.net.state_dict()
        torch.save(state_dict, os.path.join(self.ckpt_path, '{}.pth'.format(epoch)))

    def _resolve_transforms(self, aug_trans_params):
        """
            According to the given parameters, resolving transform methods
        :param aug_trans_params: the json of transform methods used
        :return: the list of augment transform methods
        """
        trans_seq = []
        for trans_name in aug_trans_params['trans_seq']:
            if trans_name == 'fixed_resize':
                resize_params = aug_trans_params['fixed_resize']
                trans_seq.append(extend_transforms.FixedResize(resize_params['size']))
            elif trans_name == 'to_tensor':
                trans_seq.append(extend_transforms.ToTensor())
            elif trans_name == 'random_horizontal_flip':
                flip_p = aug_trans_params['flip_prob']
                trans_seq.append(extend_transforms.RandomHorizontalFlip(prob=flip_p))
            elif trans_name == 'add_mask':
                mode = aug_trans_params['add_mask']["mode"]
                trans_seq.append(extend_transforms.AddMask(mode))
            elif trans_name == 'RandomErasing':
                trans_seq.append(extend_transforms.RandomErasing())
            elif trans_name == 'MaskRandomPad':
                trans_seq.append(extend_transforms.MaskRandomPad())
        return trans_seq