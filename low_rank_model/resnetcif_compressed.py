'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import OrderedDict
from model.utils import weight_decay
from operator import itemgetter
import numpy as np
__all__ = ['resnet56_compressed', 'resnet110_compressed']

class svdconv(nn.Module):
    def __init__(self, in_channel, out_channel, U, V, compression, kernel_size=3, stride=1, padding=0, scheme = "None"): # change the scheme here 
        super(svdconv, self).__init__()
        self.k = kernel_size
        self.scheme = scheme
        self.in_c = in_channel
        self.out_c = out_channel
        self.stride = stride
        self.padding = padding
        self.unfold = nn.Unfold(kernel_size=(self.k,self.k),stride=(self.stride,self.stride), padding=(self.padding,self.padding))
        self.compression_dict = compression
        U_dict={}
        V_dict = {}
        if scheme == "scheme_1":
            keys = "00"
            Utmp = U[keys]
            Vtmp = V[keys]   
            compression_tmp = compression[keys]
            if compression_tmp:
                inishape = U["inishape"] #[n,c,d1,d2]
                U_dict[keys] = nn.Parameter(torch.reshape(Utmp,(inishape[0],Utmp.shape[1],1,1))) #[n,r,1,1]
                V_dict[keys] = nn.Parameter(torch.reshape(Vtmp,(Vtmp.shape[0],inishape[1],inishape[2],inishape[3]))) #[r,c,d1,d2]
            else:
                U_dict[keys] = nn.Parameter(Utmp)
                V_dict[keys] = None
        elif scheme == "scheme_2":
            keys = "00"
            Utmp = U[keys]
            Vtmp = V[keys]   
            compression_tmp = compression[keys]  
            if compression_tmp:
                inishape = U["inishape"] #[n,c,d1,d2]
                U_dict[keys] = nn.Parameter(torch.reshape(Utmp,(inishape[0],Utmp.shape[1],1,inishape[2]))) #[n,r,1,d1]
                V_dict[keys] = nn.Parameter(torch.reshape(Vtmp,(Vtmp.shape[0],inishape[1],inishape[3],1))) #[r,c,d2,1]
            else:
                U_dict[keys] = nn.Parameter(Utmp)
                V_dict[keys] = None 
        else:
            for i in range(self.k):
                for j in range(self.k):
                    keys = str(i) + str(j)
                    Utmp = U[keys]
                    Vtmp = V[keys]
                    compression_tmp = compression[keys]
                    if compression_tmp:
                        # print(self.out_c, self.in_c)
                        # print(Vtmp.size())
                        U_dict[keys] = nn.Parameter(torch.reshape(Utmp, (self.out_c,-1 , 1, 1)))
                        V_dict[keys] = nn.Parameter(torch.reshape(Vtmp, (-1, self.in_c, 1, 1)))
                    else:
                        U_dict[keys] = nn.Parameter(torch.reshape(Utmp, (self.out_c, -1, 1, 1)))
                        V_dict[keys] = None
        self.U_weight = nn.ParameterDict(U_dict)
        self.V_weight = nn.ParameterDict(V_dict)
        # nn.init.xavier_uniform_(self.conv, gain=nn.init.calculate_gain('relu'))

    def forward(self, inp):
        k = self.k
        stride = self.stride
        inp = inp.cuda()
        h_in, w_in = inp.shape[2], inp.shape[3]

        padding = self.padding  # + k//2
        batch_size = inp.shape[0]

        h_out = (h_in + 2 * padding - (k - 1) - 1) / stride + 1
        w_out = (w_in + 2 * padding - (k - 1) - 1) / stride + 1
        try:
            h_out, w_out = h_out.int(), w_out.int()
        except:
            h_out, w_out = int(h_out), int(w_out)
        if self.scheme == "scheme_1" :
            input = inp
            keys = "00"
            Utmp = self.U_weight[keys]
            Vtmp = self.V_weight[keys]
            compression_tmp = self.compression_dict[keys]
            if compression_tmp:
                out = F.conv2d(F.conv2d(input, Vtmp,padding=padding,stride = stride), Utmp)
            else:
                out = F.conv2d(input, Utmp,padding =padding,stride = stride)
            total = torch.zeros((batch_size, self.out_c, h_out, w_out)).cuda()
            total = torch.add(out, total)
            return total
        elif self.scheme == "scheme_2":
            input = inp
            keys = "00"
            Utmp = self.U_weight[keys]
            Vtmp = self.V_weight[keys]
            compression_tmp = self.compression_dict[keys]
            if compression_tmp:
                out = F.conv2d(F.conv2d(input, Vtmp,padding=padding,stride = stride), Utmp,stride = stride)
            else:
                out = F.conv2d(input, Utmp,padding =padding,stride = stride)
            total = torch.zeros((batch_size, self.out_c, h_out, w_out)).cuda()
            total = torch.add(out, total)
            return total
        else:
            input = self.unfold(inp)
            input = input.view(-1, self.in_c, k, k, h_out, w_out)
            total = torch.zeros((batch_size, self.out_c, h_out, w_out)).cuda()
            for i in range(k):
                for j in range(k):
                    keys = str(i) + str(j)
                    Utmp = self.U_weight[keys]
                    Vtmp = self.V_weight[keys]
                    compression_tmp = self.compression_dict[keys]
                    if compression_tmp:
                        out = F.conv2d(input[:, :, i, j, :, :], Vtmp)
                        out = F.conv2d(out, Utmp)
                    else:
                        out = F.conv2d(input[:, :, i, j, :, :], Utmp)
                    total = torch.add(out, total)
            return total

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A',sub_compression_dict=[]):
        super(BasicBlock, self).__init__()
        U = sub_compression_dict[0]["U"]
        V = sub_compression_dict[0]["V"]
        compression = sub_compression_dict[0]["compression"]
        if any(compression.values()):
            self.compressible_conv1 = svdconv(in_planes, planes, U, V, compression, kernel_size=3, stride=stride, padding=1)
        else:
            self.compressible_conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        U = sub_compression_dict[1]["U"]
        V = sub_compression_dict[1]["V"]
        compression = sub_compression_dict[1]["compression"]
        if any(compression.values()):
            self.compressible_conv2 = svdconv(planes, planes, U, V, compression, kernel_size=3, stride=1, padding=1)
        else:
            self.compressible_conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.option = option
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(OrderedDict([
                    ('compressible_conv2d', nn.Conv2d(in_planes, self.expansion * planes, kernel_size=2, stride=stride, bias=False)),
                    ('batch_norm', nn.BatchNorm2d(self.expansion * planes))
                ]))

    def forward(self, x):
        out = F.relu(self.bn1(self.compressible_conv1(x)))
        # print('after compressible_conv1:', out.size(), self.compressible_conv1)
        out = self.bn2(self.compressible_conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, option='A',compression_task={}):
        super(ResNet, self).__init__()
        self.in_planes = 16

        layer_key = 'compressible_conv1'
        value = compression_task[layer_key]
        U = value["U"]
        V = value["V"]
        compression = value["compression"]
        if any(compression.values()):
            self.compressible_conv1 = svdconv(3, 16, U, V, compression, kernel_size=3, stride=1, padding=1)
        else:
            self.compressible_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        layer1_dict = {}
        layer2_dict = {}
        layer3_dict = {}
        for key,value in compression_task.items():
            if 'layer1.' in key:
                layer1_dict[key] = value
            if 'layer2.' in key:
                layer2_dict[key] = value
            if 'layer3.' in key:
                layer3_dict[key] = value

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, option=option,\
                                       compression_dict=layer1_dict,submodule_num=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, option=option,\
                                       compression_dict=layer2_dict,submodule_num=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, option=option,\
                                       compression_dict=layer3_dict, submodule_num=3)
        self.compressible_linear = nn.Linear(64, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.weight_decay = weight_decay(self)
        self.loss = lambda x, target: nn.CrossEntropyLoss()(x, target) + 1e-4*self.weight_decay()

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, option,compression_dict={},submodule_num=0):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        i = 0
        for stride in strides:
            submodule_key = ['layer' + str(submodule_num) + '.' + str(i) + '.compressible_conv1','layer' + str(submodule_num) + '.' + str(i) + '.compressible_conv2']
            sub_compression_dict = itemgetter(*submodule_key)(compression_dict)
            layers.append(block(self.in_planes, planes, stride, option, sub_compression_dict))
            self.in_planes = planes * block.expansion
            i = i + 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.compressible_conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.compressible_linear(x)
        return x

    def compressible_modules(self):
        for name, module in self.named_modules():
            if 'compressible' in name and name not in self.except_:
                yield name, module


# def resnetcif20():
#     return ResNet(BasicBlock, [3, 3, 3], option='A')
#
# def resnetcif20b():
#     return ResNet(BasicBlock, [3, 3, 3], option='B')
#
# def resnetcif32():
#     return ResNet(BasicBlock, [5, 5, 5])
#
#
# def resnetcif44():
#     return ResNet(BasicBlock, [7, 7, 7])


def resnet56_compressed(compression_task={}):
    return ResNet(BasicBlock, [9, 9, 9], compression_task=compression_task)


def resnet110_compressed(compression_task={}):
    return ResNet(BasicBlock, [18, 18, 18], compression_task=compression_task)


# def resnetcif1202():
#     return ResNet(BasicBlock, [200, 200, 200])
