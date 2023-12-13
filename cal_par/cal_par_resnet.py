import torch
import numpy as np

pardict =torch.load('resnet56_r_0.1.th',map_location='cpu')
pardict = pardict['model_state']
par = 0
flops = 0
key_pardict = pardict.keys()
for i in key_pardict:
    if 'output' not in i:
        layer_weight = pardict[i]
        par = par + np.prod(layer_weight.size())
#print(key_pardict)



print(par/1000**2)

par = 0
for i in key_pardict:
    if 'compressible_conv1' in i and 'layer' not in i:
        weight = pardict[i]
        [a,b,_,_] = weight.size()
        flops = flops + a*b*32*32
        par = par + a*b     
    elif 'layer1' in i and 'compressible' in i and '_weight' in i:
        weight = pardict[i]
        [a, b, _, _] = weight.size()
        flops = flops + a*b*32*32
        par = par + a*b
    elif 'layer1' in i and ('compressible_conv2.weight' in i or 'compressible_conv1.weight' in i):
        weight = pardict[i]
        [a, b, _, _] = weight.size()
        flops = flops + a*b*3*3*32*32
        par = par + a*b*3*3
    elif 'layer2.0.compressible_conv1.weight' in i:
        weight = pardict[i]
        [a, b, _, _] = weight.size()
        flops = flops + a * b * 3 * 3 * 16 * 16
        par = par + a * b * 3 * 3
    elif 'layer2' in i and 'compressible' in i and '_weight' in i:
        weight = pardict[i]
        [a, b, _, _] = weight.size()
        flops = flops + a*b*16*16
        par = par + a*b
    elif 'layer2' in i and ('compressible_conv2.weight' in i or 'compressible_conv1.weight' in i):
        weight = pardict[i]
        [a, b, _, _] = weight.size()
        flops = flops + a*b*3*3*16*16
        par = par + a*b*3*3
    elif 'layer3.0.compressible_conv1.weight' in i:
        weight = pardict[i]
        [a, b, _, _] = weight.size()
        flops = flops + a * b * 3 * 3 * 8 * 8
        par = par + a * b * 3 * 3
    elif 'layer3' in i and 'compressible' in i and '_weight' in i:
        weight = pardict[i]
        [a, b, _, _] = weight.size()
        flops = flops + a * b * 8 * 8
        par = par + a * b 
    elif 'layer3' in i and ('compressible_conv2.weight' in i or 'compressible_conv1.weight' in i):
        weight = pardict[i]
        [a, b, _, _] = weight.size()
        flops = flops + a * b * 3 * 3 * 8 * 8
        par = par + a * b * 3 * 3
    else:
        layer_weight = pardict[i]
        try:
            par = par + layer_weight.size()[0]
        except:
            par = par + 1

par = par + 64*10
flops = flops + 64*10


print(par/1000**2)
org = 0.85
print(1-par/1000**2/org)

print(flops/1000**2)