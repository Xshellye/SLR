#!/usr/bin/env python3
import numpy as np
from scipy.linalg import svd
import torch


def tensor_to_matrix(tensor, conv_scheme='scheme_1'):
    """
    Reshape a tensor into a matrix according to a scheme.
    :param tensor:
    :param conv_scheme: currently we support two scheme: scheme-1 and scheme-2
    :return: a matrix containing tensor's elements
    """
    matrix = tensor
    init_shape = tensor.shape
    if tensor.ndim == 4:
        if conv_scheme == 'scheme_1':
            matrix = tensor.reshape(init_shape[0], -1)
        elif conv_scheme == 'scheme_2':
            [n, m, d1, d2] = tensor.shape
            swapped = tensor.swapaxes(1, 2)
            matrix = swapped.reshape([n * d1, m * d2])
        else:
            pass

    #if not np.allclose(tensor, matrix_to_tensor(matrix, init_shape, conv_scheme)):
    #    raise NotImplementedError("The Tensor->Matrix->Tensor conversion is not correctly implemented.")
    return matrix


def matrix_to_tensor(matrix, init_shape, conv_scheme='scheme_1'):
    """
    Reshapes previously reshaped matrix into the original tensor.
    :param matrix: matrix to be converted back
    :param init_shape: the initial shape of the tensor
    :param conv_scheme: the convolutional scheme used for reshape
    :return:
    """
    tensor = matrix
    if len(init_shape) == 4:
        if conv_scheme == "scheme_1":
            tensor = matrix.reshape(init_shape)
        elif conv_scheme == 'scheme_2':
            [n, m, d1, d2] = init_shape
            tensor = matrix.reshape([n, d1, m, d2]).swapaxes(1, 2)
        else:
            raise NotImplementedError('This type of shape parametrization is not implemented yet.')
    return tensor

class RankSelection():
    def __init__(self, module_name, ratio=0.05, scheme = 'None'):
        self.module_name = module_name
        self.selected_rank = []
        self.reconstruction_error_ratio = ratio
        self.conv_scheme = scheme

    def compress(self, data):
        init_shape = data.shape
        if self.conv_scheme != 'None':           
            if data.ndim == 2:
                matrix = data
            elif data.ndim == 4:
                matrix = tensor_to_matrix(data, self.conv_scheme)
            U, sigma, V = torch.linalg.svd(matrix, full_matrices=False)
            Ucut_dict = {}
            Vcut_dict = {}
            compression = {}
            max_rank = min(init_shape[0],init_shape[1])
            selected_rank = max_rank
            sigma2 = sigma[:] 
            sigma2_tmp = sigma2.cpu().detach().numpy()
            rank_i_diff_frb_norm_sq = (sigma2_tmp[::-1]**2).cumsum()[::-1]
            frob_norm_sq = rank_i_diff_frb_norm_sq[0]
            norm_sq = frob_norm_sq
            # 搜索满足重构误差标准的最高秩。
            for r in range(max_rank-1,0,-1):
                reconstruction_error_value = rank_i_diff_frb_norm_sq[r]*(1/norm_sq)
                if reconstruction_error_value < self.reconstruction_error_ratio and r < selected_rank:
                    selected_rank = r
            
            if selected_rank*init_shape[0] + selected_rank*init_shape[1] < init_shape[0] * init_shape[1]:
                diag = torch.diag_embed((sigma2[:selected_rank])**0.5)
                Ucut = torch.matmul(U[:,:selected_rank],diag)
                Vcut = torch.matmul(diag,V[:selected_rank,:])
                # if selected_rank <= max_rank:
                #     frob_norm_diff = (rank_i_diff_frb_norm_sq[selected_rank]/norm_sq if selected_rank < max_rank else 0)
                # else:
                #     frob_norm_diff = 0
                self.selected_rank.append(selected_rank)
                # change key to '00' ，corresponding scheme class svdconv in resnetcif_compressed.py should apply the same change
                # this keys only for scheme 1 and 2
                keys = "00"
                Ucut_dict[keys] = Ucut
                Ucut_dict["inishape"] = init_shape
                Vcut_dict[keys] = Vcut
                compression[keys] = True
            else:
                keys = "00"
                Ucut_dict[keys] = data
                Vcut_dict[keys] = None
                compression[keys] = False
                self.selected_rank.append(selected_rank)
        else:
            try:
                A = data.permute(2, 3, 0, 1)
            except:
                A = torch.unsqueeze(data,dim=2)
                A = torch.unsqueeze(A,dim=2)
                init_shape = A.shape
                A = A.permute(2, 3, 0, 1)
            
            U, sigma, V = torch.linalg.svd(A, full_matrices=False)
            U = U.permute(2, 3, 0, 1)
            sigma = sigma.permute(2, 0, 1)
            V = V.permute(2, 3, 0, 1)
            Ucut_dict = {}
            Vcut_dict = {}
            compression = {}
            maxrank = []
            for i in range(init_shape[2]):
                for j in range(init_shape[3]):
                    max_rank = min(init_shape[0],init_shape[1])
                    selected_rank = max_rank
                    sigma2 = sigma[:, i, j]  # sigular value
                    sigma2_tmp = sigma2.cpu().detach().numpy()
                    rank_i_diff_frb_norm_sq = (sigma2_tmp[::-1]**2).cumsum()[::-1]
                    frob_norm_sq = rank_i_diff_frb_norm_sq[0]
                    norm_sq = frob_norm_sq
                    maxrank.append(max_rank)
                    for r in range(max_rank-1,0,-1):
                        reconstruction_error_value = rank_i_diff_frb_norm_sq[r]*(1/norm_sq)
                        if reconstruction_error_value < self.reconstruction_error_ratio and r < selected_rank:
                            selected_rank = r
                    self.selected_rank.append(selected_rank)       

                    if selected_rank*init_shape[0]+selected_rank*init_shape[1] < init_shape[0]*init_shape[1]:
                        diag = torch.diag_embed((sigma2[:selected_rank])**0.5)
                        Ucut = torch.matmul(U[:,:selected_rank,i,j],diag)
                        Vcut = torch.matmul(diag,V[:selected_rank,:,i,j])
                        # if selected_rank <= max_rank:
                        #     frob_norm_diff = (rank_i_diff_frb_norm_sq[selected_rank]/norm_sq if selected_rank < max_rank else 0)
                        # else:
                        #     frob_norm_diff = 0
                        #self.selected_rank.append(selected_rank)
                        keys = str(i) + str(j)
                        Ucut_dict[keys] = Ucut
                        Vcut_dict[keys] = Vcut
                        compression[keys] = True  
                    else:
                        print(self.module_name + "not compressed")
                        keys = str(i) + str(j)
                        Ucut_dict[keys] = data[:,:,i,j]
                        Vcut_dict[keys] = None
                        compression[keys] = False
                        
                        
        state_dict = {"U": Ucut_dict, "V": Vcut_dict, "compression": compression}
        print("{},selectedrank:{}".format(self.module_name, sum(map(int,self.selected_rank))))
        #print("maxrank:%s" % sum(map(int,maxrank)))
        return state_dict
