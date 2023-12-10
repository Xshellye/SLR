import torch
import numpy as np

pardict =torch.load('resnet56_ft_0.4_epoch83_0.9332.th',map_location='cpu')
pardict = pardict['model_state']
par = 0
flops = 0
key_pardict = pardict.keys()
for i in key_pardict:
    if 'output' not in i:
        layer_weight = pardict[i]
        par = par + np.prod(layer_weight.size())

# print(par)
print(par/1000**2)
par = 0
for i in key_pardict:
    if 'compressible_conv1.U' in i and 'layer' not in i:
        weight = pardict[i]
        [a,b,_,_] = weight.size()
        if b < 3:
            flops = flops + (a*b+b*3)*32*32
            par = par + a*b + b*3
        else:
            flops = flops + a*b*32*32
            par = par + a*b
    if 'layer1' in i and 'compressible' in i and 'U_weight' in i:
        weight = pardict[i]
        [a, b, _, _] = weight.size()
        if b < 16:
            flops = flops + (a*b+b*16)*32*32
            par = par + a*b + b*16
        else:
            flops = flops + a*b*32*32
            par = par + a*b
    if 'layer1' in i and ('compressible_conv2.weight' in i or 'compressible_conv1.weight' in i):
        weight = pardict[i]
        [a, b, _, _] = weight.size()
        flops = flops + a*b*3*3*32*32
        par = par + a*b*3*3
    if 'layer2.0.compressible_conv1.weight' in i:
        weight = pardict[i]
        [a, b, _, _] = weight.size()
        flops = flops + a * b * 3 * 3 * 16 * 16
        par = par + a * b * 3 * 3
    else:
        if 'layer2.0.compressible_conv1' in i and 'U_weight' in i:
            weight = pardict[i]
            [a, b, _, _] = weight.size()
            if b < 16:
                flops = flops + (a*b+b*16)*16*16
                par = par + a*b + b*16
            else:
                flops = flops + a*b*16*16
                par = par + a*b
        else:
            if 'layer2' in i and 'compressible' in i and 'U_weight' in i:
                weight = pardict[i]
                [a, b, _, _] = weight.size()
                if b < 32:
                    flops = flops + (a*b+b*32)*16*16
                    par = par + a*b + b*32
                else:
                    flops = flops + a*b*16*16
                    par = par + a*b
            if 'layer2' in i and ('compressible_conv2.weight' in i or 'compressible_conv1.weight' in i):
                weight = pardict[i]
                [a, b, _, _] = weight.size()
                flops = flops + a*b*3*3*16*16
                par = par + a*b*3*3
    if 'layer3.0.compressible_conv1.weight' in i:
        weight = pardict[i]
        [a, b, _, _] = weight.size()
        flops = flops + a * b * 3 * 3 * 8 * 8
        par = par + a * b * 3 * 3
    else:
        if 'layer3.0.compressible_conv1' in i and 'U_weight' in i:
            weight = pardict[i]
            [a, b, _, _] = weight.size()
            if b < 32:
                flops = flops + (a * b + b * 32) * 8 * 8
                par = par + a * b + b * 32
            else:
                flops = flops + a * b * 8 * 8
                par = par + a * b
        else:
            if 'layer3' in i and 'compressible' in i and 'U_weight' in i:
                weight = pardict[i]
                [a, b, _, _] = weight.size()
                if b < 32:
                    flops = flops + (a * b + b * 64) * 8 * 8
                    par = par + a * b + b * 64
                else:
                    flops = flops + a * b * 8 * 8
                    par = par + a * b
            if 'layer3' in i and ('compressible_conv2.weight' in i or 'compressible_conv1.weight' in i):
                weight = pardict[i]
                [a, b, _, _] = weight.size()
                flops = flops + a * b * 3 * 3 * 8 * 8
                par = par + a * b * 3 * 3

par = par + 64*10
flops = flops + 64*10
print(par/1000**2)
print(flops/1000**2)