import torch
import torch.nn as nn
import torch.nn.parallel
from torchvision import models
import torch.utils.model_zoo as model_zoo
import pdb
from .nmn_block import EpsStep

class InceptionV3(nn.Module):
    def __init__(self, c, num_classes):
        super(InceptionV3, self).__init__()
        self.conv1 = nn.Conv2d(c, 3, kernel_size=2, stride=1, padding=22, bias=True)
        self.model = models.inception_v3(pretrained=True)
        self.model.fc = nn.Linear(2048, num_classes)
        self.model.aux_logits = False
    
    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.model(out)
        return out

class EfficientnetB3(nn.Module):
    def __init__(self, c, num_classes):
        super(EfficientnetB3, self).__init__()
        self.conv1 = nn.Conv2d(c, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(1536, num_classes)
        print(self.model.classifier)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.model(out)
        return out

class EfficientnetB5(nn.Module):
    def __init__(self, c, num_classes):
        super(EfficientnetB5, self).__init__()
        self.conv1 = nn.Conv2d(c, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.model = models.efficientnet_b5(pretrained=True)
        self.model.classifier[1] = nn.Linear(2048, num_classes)
        print(self.model.classifier)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.model(out)
        return out


class EpsNet(torch.nn.Module):
    def __init__(self,i_size=2048,h_size=1024,o_size=3,eps=0.1, T=10):
        super(EpsNet,self).__init__()
        self.i_size=i_size
        self.h_size=h_size
        self.o_size=o_size
        self.eps=eps
        self.T = T
        self.l1=torch.nn.Linear(i_size,h_size,bias=True)
        self.l2=torch.nn.Linear(h_size,o_size,bias=True)
    def forward(self,i):
        x = self.l1(i)
        x = x.view(x.size(0), self.h_size)
        for i in range(self.T):
            if i==0:
                y = self.eps * torch.sin(x) * torch.sin(x)     
            else:
                y = self.eps * torch.sin(x+y) * torch.sin(x+y)     
        return self.l2(y)

class InceptionEpsNet(nn.Module):
    def __init__(self, c, num_classes):
        super(InceptionEpsNet, self).__init__()
        self.conv1 = nn.Conv2d(c, 3, kernel_size=2, stride=1, padding=22, bias=True)
        self.model = models.inception_v3(pretrained=True)
        self.model.fc = nn.Linear(2048, 1024)
        self.epsstep = EpsStep()
        self.outlayer = nn.Linear(1024, num_classes)
        self.model.aux_logits = False
    
    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.model(out)
        y = self.epsstep(out)
        z = self.outlayer(y)
        return z


class EfficientnetB3EpsNet(nn.Module):
    def __init__(self, c, num_classes):
        super(EfficientnetB3EpsNet, self).__init__()
        self.conv1 = nn.Conv2d(c, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.model = models.efficientnet_b3(pretrained=True)
        self.model.classifier[1] = nn.Linear(1536, 1024)
        self.epsstep = EpsStep()
        self.outlayer = nn.Linear(1024, num_classes)
    
    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.model(out)
        y = self.epsstep(out)
        z = self.outlayer(y)
        return z


class MobileNetV3(nn.Module):
    def __init__(self, c, num_classes):
        super(MobileNetV3, self).__init__()
        self.conv1 = nn.Conv2d(c, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.model = models.mobilenet_v3_small(pretrained=True)
        self.model.classifier[3] = nn.Linear(1024, num_classes)
        print(self.model.classifier)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.model(out)        
        return out

class ShuffleNetV2(nn.Module):
    def __init__(self, c, num_classes):
        super(ShuffleNetV2, self).__init__()
        self.conv1 = nn.Conv2d(c, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.model = models.shufflenet_v2_x1_0(pretrained=True)
        self.model.fc = nn.Linear(1024, num_classes)
        print(self.model.fc)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.model(out)        
        return out

class SqueezeNet(nn.Module):
    def __init__(self, c, num_classes):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(c, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.model = models.squeezenet1_1(pretrained=True)
        self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0,bias=True)
        print(self.model.classifier)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.model(out)        
        return out