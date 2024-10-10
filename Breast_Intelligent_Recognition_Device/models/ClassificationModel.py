import torch

from .odeblock import NMODEBlock, ODEBlock
from .odefuncs import *

from torch import nn
from torch import autograd


class ClassificationModel(nn.Module):
    def __init__(self, conf, model_name='BaseODEFunc', num_class=10, **kwargs):
        super(ClassificationModel, self).__init__()
        self.model_name = model_name
        self.num_hidden = conf["num_hidden"]
        
        self.net = NMODEBlock(NMODEFunc(conf["in_features"], conf["num_hidden"]), backodefunc=NMODEFuncBack(conf["in_features"], conf["num_hidden"]))
        self.fc = nn.Linear(conf["num_hidden"], num_class)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
    def backward(self, J, y, x):
        # print(J.shape, J, "-&"*10)
        lambdaT = autograd.grad(J, y, grad_outputs=None, retain_graph=True, create_graph=False, only_inputs=True, allow_unused=False)[0]
        self.fc.weight.grad = autograd.grad(J, self.fc.weight, grad_outputs=None, retain_graph=True, create_graph=False, only_inputs=True, allow_unused=False)[0]
        self.fc.bias.grad = autograd.grad(J, self.fc.bias, grad_outputs=None, retain_graph=True, create_graph=False, only_inputs=True, allow_unused=False)[0]
        _, _, gradw = self.net.backward(y.detach(), lambdaT.detach(), torch.zeros_like(self.net.odefunc.w.weight), x.detach())
        # print(self.net.odefunc.w.weight.shape, gradw.shape, "-&"*10)
        self.net.odefunc.w.weight.grad = gradw
        
        
    def forward(self, x, require_y=False):
        bs, nf = x.size()
        y = self.net(x)
        out = self.fc(y)
        if require_y:
            return out, y
        else:
            return out


