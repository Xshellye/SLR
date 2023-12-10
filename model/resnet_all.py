from model.resnetcif import resnetcif56, resnetcif110
from utils import AverageMeter, Recorder, format_time, data_loader, compute_acc_loss
import torch
import low_rank_model as low_rank_model_def
import time
from torch import nn
from compression_type.low_rank import RankSelection


def update_dict(pre_model_dict, model_dict):
  state_dict = {k: v for k, v in pre_model_dict.items() if k in model_dict.keys()}
  model_dict.update(state_dict)
  return model_dict

class resnet_all():
  def __init__(self, name, model, stored_ref):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.name = name
    self.model = model.to(self.device)
    self.train_loader, self.test_loader = data_loader(batch_size=128, n_workers=4, dataset="CIFAR10")
    pretrained_model = torch.load('references/{}.th'.format(stored_ref),map_location='cpu')
    self.model.load_state_dict(pretrained_model)
    self.flops_subtract_adv = 0
    self.params_subtract_adv = 0
    self.parametrization = 'simple'

#select rank for each layer
  def create_lr_compression_task(self,ratio,scheme = None,compress_state=True):

    if compress_state:
        compression_tasks = {}
        for i, (w_get, module_name) in enumerate([((lambda x=x: getattr(x, 'weight')), name) for name, x in self.model.named_modules() if 'compressible_conv' in name]):
            compression_tasks[module_name] \
            = RankSelection(module_name=module_name,ratio=ratio,scheme = scheme).compress(
            w_get())
        torch.save(compression_tasks,f'result_compression/{self.name}_ratio_{ratio}.th')
    else:
        compression_tasks = torch.load(f'result_compression/{self.name}_ratio_{ratio}.th')
    return compression_tasks

  def compression_evaluate(self,ratio,scheme = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre_model = self.model
    state_dict = pre_model.state_dict()
    compressed_model = getattr(low_rank_model_def, self.name + '_compressed')(self.create_lr_compression_task(ratio=ratio,scheme=scheme,compress_state=True))
    net = compressed_model.to(device)
    net.load_state_dict(update_dict(state_dict, net.state_dict()))
    print("Low rank layers of the model has been successfully reparameterized with sequence of full-rank matrices.")

    # def my_forward_eval(x, target):
    #   out_ = net.forward(x)
    #   return out_, net.loss(out_, target)

    # net.eval()
    # accuracy_train, ave_loss_train = compute_acc_loss(my_forward_eval, self.train_loader)
    # print('\tBefore finetuning, the train loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss_train, accuracy_train))
    # # rec.record('train_nested', [ave_loss, accuracy, training_time, step + 1])
    # accuracy_test, ave_loss_test = compute_acc_loss(my_forward_eval, self.test_loader)
    # # self.model.train()
    # print('\tBefore finetuning, the test loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss_test, accuracy_test))

    return net

class resnet56(resnet_all):
  def __init__(self):
    super(resnet56, self).__init__("resnet56", resnetcif56(), 'resnet56')


class resnet110(resnet_all):
  def __init__(self):
    super(resnet110, self).__init__("resnet110", resnetcif110(), 'resnetcif110')
