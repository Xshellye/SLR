import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math

__all__ = ['densenet40_compressed']

class svdconv(nn.Module):
    def __init__(self, in_channel, out_channel, U, V, compression, kernel_size=3, stride=1, padding=0, fc=False):
        super(svdconv, self).__init__()
        self.k = kernel_size
        self.in_c = in_channel
        self.out_c = out_channel
        self.stride = stride
        self.fc = fc
        self.padding = padding
        self.unfold = nn.Unfold(kernel_size=(self.k,self.k),stride=(self.stride,self.stride), padding=(self.padding,self.padding))
        self.compression_dict = compression
        U_dict={}
        V_dict = {}
        if fc:
            keys = '00'
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
        h_in, w_in = inp.shape[2], inp.shape[3]

        padding = self.padding  # + k//2
        batch_size = inp.shape[0]

        h_out = (h_in + 2 * padding - (k - 1) - 1) / stride + 1
        w_out = (w_in + 2 * padding - (k - 1) - 1) / stride + 1
        try:
            h_out, w_out = h_out.int(), w_out.int()
        except:
            h_out, w_out = int(h_out), int(w_out)

        input = self.unfold(inp)
        input = input.view(-1, self.in_c, k, k, h_out, w_out)
        total = torch.zeros((batch_size, self.out_c, h_out, w_out),device="cuda")
        for i in range(k):
            for j in range(k):
                keys = str(i) + str(j)
                Utmp = self.U_weight[keys]
                Vtmp = self.V_weight[keys]
                compression_tmp = self.compression_dict[keys]
                if compression_tmp:
                    out = F.conv2d(F.conv2d(input[:, :, i, j, :, :], Vtmp), Utmp)
                else:
                    out = F.conv2d(input[:, :, i, j, :, :], Utmp)

                total = torch.add(out, total)
        return total
    
class Bottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0, sub_compression_dict=[]):
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        U = sub_compression_dict[0]["U"]
        V = sub_compression_dict[0]["V"]
        compression = sub_compression_dict[0]["compression"]
        if any(compression.values()):
            self.compressible_conv1 = svdconv(inplanes, planes, U, V, compression, kernel_size=1)
        else:
            self.compressible_conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            
        self.bn2 = nn.BatchNorm2d(planes)
        U = sub_compression_dict[1]["U"]
        V = sub_compression_dict[1]["V"]
        compression = sub_compression_dict[1]["compression"]
        if any(compression.values()):
            self.compressible_conv2 = svdconv(planes, growthRate, U, V, compression, kernel_size=3, stride=1, padding=1)
        else:
            self.compressible_conv2 = nn.Conv2d(planes, growthRate, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.compressible_conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.compressible_conv2(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0, sub_compression_dict={}):
        super(BasicBlock, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        U = sub_compression_dict["U"]
        V = sub_compression_dict["V"]
        compression = sub_compression_dict["compression"]
        if any(compression.values()):
            self.compressible_conv1 = svdconv(inplanes, growthRate, U, V, compression, kernel_size=3, padding=1)
        else:
            self.compressible_conv1 = nn.Conv2d(inplanes, growthRate, kernel_size=3, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.compressible_conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes,com_dict={}):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        U = com_dict["U"]
        V = com_dict["V"]
        compression = com_dict["compression"]
        if any(compression.values()):
            self.compressible_conv1 = svdconv(inplanes, outplanes, U, V, compression, kernel_size=1)
        else:
            self.compressible_conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.compressible_conv1(out)
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet(nn.Module):

    def __init__(self, depth=22, block=Bottleneck, dropRate=0, num_classes=10, growthRate=12, compressionRate=2,compression_task={}):
        super(DenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3 if block == BasicBlock else (depth - 4) // 6

        self.growthRate = growthRate
        self.dropRate = dropRate
        self.loss = lambda x, target: nn.CrossEntropyLoss()(x, target)
        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = growthRate * 2
        
        layer_key = 'compressible_conv1'
        value = compression_task[layer_key]
        U = value["U"]
        V = value["V"]
        compression = value["compression"]
        if any(compression.values()):
            self.compressible_conv1 = svdconv(3, self.inplanes, U, V, compression, kernel_size=3, padding=1)
        else:
            self.compressible_conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
    
        dense1_dict={}
        dense2_dict={}
        dense3_dict={}
        trans1_dict={} 
        trans2_dict={}
        for key,value in compression_task.items():
            if 'dense1.' in key:
                dense1_dict[key] = value
            if 'dense2.' in key:
                dense2_dict[key] = value
            if 'dense3.' in key:
                dense3_dict[key] = value
            if 'trans1.' in key:
                trans1_dict[key] = value
            if 'trans2.' in key:
                trans2_dict[key] = value
        
        
        self.dense1 = self._make_denseblock(block, n, \
                                            compression_dict=dense1_dict,submodule_num=1)
        self.trans1 = self._make_transition(compressionRate,\
                                            compression_dict=trans1_dict,submodule_num=1)
        self.dense2 = self._make_denseblock(block, n, \
                                            compression_dict=dense2_dict,submodule_num=2)
        self.trans2 = self._make_transition(compressionRate,\
                                            compression_dict=trans2_dict,submodule_num=2)
        self.dense3 = self._make_denseblock(block, n, \
                                            compression_dict=dense3_dict,submodule_num=3)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        
        
        layer_key = 'compressible_fc'
        value = compression_task[layer_key]
        U2 = value["U"]
        V2 = value["V"]
        compression2 = value["compression"]
        if any(compression2.values()):
            self.compressible_fc = svdconv(self.inplanes, num_classes,U2, V2,compression2,kernel_size=1,fc=True)
        else:
            self.compressible_fc = nn.Linear(self.inplanes, num_classes)
            
    def _make_denseblock(self, block, blocks, compression_dict={}, submodule_num=0):
        layers = []
        for i in range(blocks):
            sub_compression_dict = compression_dict['dense' + str(submodule_num) + '.' + str(i) + '.compressible_conv1']
            layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate,sub_compression_dict = sub_compression_dict))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate,compression_dict={},submodule_num=0):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        compression_dict = compression_dict['trans' + str(submodule_num) + '.compressible_conv1']
        self.inplanes = outplanes
        return Transition(inplanes, outplanes,com_dict=compression_dict)

    def forward(self, x):
        x = x.cuda()
        x = self.compressible_conv1(x)
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)       
        x = self.compressible_fc(x) 
        #x = x.view(x.size(0), -1)   
        x = torch.squeeze(x)
        return x



def densenet40_compressed(compression_task={}):
    model = DenseNet(block=BasicBlock, depth=40, dropRate=0, num_classes=10, growthRate=12, compressionRate=1,compression_task=compression_task)
    return model
