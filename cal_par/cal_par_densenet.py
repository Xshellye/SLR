import torch
import numpy as np

pardict =torch.load('densenet40_ft_0.19_epoch41_0.9436.th',map_location='cpu')["model_state"]
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
    if 'dense' not in i and 'trans' not in i and 'compressible_conv1' in i and ('U_weight' in i or 'V_weight'in i) :
        weight = pardict[i]
        [a, b, _, _] = weight.size()
        flops = flops + a*b*32*32
        par = par + a*b
        
    elif 'dense1' in i and 'compressible' in i:
        weight = pardict[i]
        [a, b, _, _] = weight.size()
        flops = flops + a*b*32*32
        par = par + a*b       
    elif 'trans1' in i and 'compressible_conv1' in i:
        weight = pardict[i]
        [a, b, _, _] = weight.size()
        flops = flops + a*b*32*32
        par = par + a*b
        
    elif 'dense2' in i and 'compressible' in i:
        weight = pardict[i]
        [a, b, _, _] = weight.size()
        flops = flops + a*b*16*16
        par = par + a*b
     
    elif 'trans2' in i and 'compressible_conv1' in i:
        weight = pardict[i]
        [a, b, _, _] = weight.size()
        flops = flops + a*b*16*16
        par = par + a*b
        
    elif 'dense3' in i and 'compressible' in i :
        weight = pardict[i]
        [a, b, _, _] = weight.size()
        flops = flops + a*b*8*8
        par = par + a*b
    elif 'fc' in i and 'compressible' in i:
        weight = pardict[i]
        [a, b, _, _] = weight.size()
        flops = flops + a*b
        par = par + a*b
    else:
        layer_weight = pardict[i]
        try:
            par = par + layer_weight.size()[0]
        except:
            par = par + 1


print(par/1000**2)
print(flops/1000**2)