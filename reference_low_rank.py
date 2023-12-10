#!/bin/env/python3
from types import ModuleType
import model as model_def
from utils import get_number_of_param,data_loader,AverageMeter,format_time,compute_acc_loss
import argparse
import torch
from torch import optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import time
import os


if __name__ == '__main__':
    if not os.path.exists('result'):
        os.makedirs('result')

    model_names = model_def.__all__
    parser = argparse.ArgumentParser(description='Reference Network Trainer for MNIST and CIFAR10 networks')
    parser.add_argument('--arch', '-a', metavar='ARCH', default="resnet56",
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                             ' (default: {})'.format(model_names[0]))
    parser.add_argument('--dataset', choices=['MNIST', 'CIFAR10'], default='CIFAR10')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--checkpoint', type=int, default=20)
    parser.add_argument('--epochs', default=110, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.09, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--scheduler', choices=['exponential', 'steps'], default='steps')
    parser.add_argument('--milestones', nargs='+', default=[60,80,90,100], type=int)
    parser.add_argument('--lr_decay', type=float, default=0.1)

    parser.add_argument('--print_freq', '-p', default=2, type=int,
                        metavar='N', help='print frequency (default: 2)')
    parser.add_argument('--resume', action='store_true',
                        help='resumes from recent checkpoint')
    parser.add_argument('--ratio','-r',default = 0.7,type=float)
    parser.add_argument('--scheme',default = "None", type=str, help = "scheme_1 or scheme_2 or None")
    args = parser.parse_args()
    print(args)
    para_infor = []
    Flops_infor = []
    # cudnn.benchmark = True
    # uncompressed model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(model_def, args.arch)()
    original_model = model.model
    # original_model = original_model.to(device)
    train_loader, test_loader = data_loader(batch_size=args.batch_size, n_workers=args.workers,dataset=args.dataset)

    # def my_forward_eval(x, target):
    #     out_ = original_model.forward(x)
    #     return out_, original_model.loss(out_, target)
    # original_model.eval()
    # accuracy_train, ave_loss_train = compute_acc_loss(my_forward_eval, train_loader)
    # print('the train loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss_train, accuracy_train))
    # accuracy_test, ave_loss_test = compute_acc_loss(my_forward_eval, test_loader)
    # print('the test loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss_test, accuracy_test))
    
####################################################################################################################

    # compressed network evaluation without finetuning
    if args.arch == 'resnet56':
        net = model.compression_evaluate(args.ratio, args.scheme)
    else:
        net = model.compression_evaluate(args.ratio)

    torch.save(net.state_dict(), f'result/{args.arch}_lr_{args.ratio}.th')
    
    
    #fine-tuning
    net = net.to(device)
    all_start_time = time.time()
    start_epoch = 0
    my_params = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = torch.optim.SGD(my_params, args.lr,
                                  momentum=args.momentum, nesterov=True)
    epoch_time = AverageMeter()
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=args.lr_decay, last_epoch=start_epoch-1)

    def my_eval(x, target):
        out_ = net.forward(x)
        return out_, net.loss(out_, target)
    train_info = {}
    test_info = {}
    bestacc = 0
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        avg_loss_ = AverageMeter()
        for x, target in train_loader:
            optimizer.zero_grad()
            x, target = x.cuda(), target.cuda()
            out = net.forward(x)
            loss = net.loss(out, target)
            loss.backward()
            avg_loss_.update(loss.item())
            optimizer.step()
        end_time = time.time()
        training_time = end_time - all_start_time
        epoch_time.update(end_time - start_time)

        print("Epoch {0} finished in {1.val:.3f}s (avg: {1.avg:.3f}s). Training for {2}"
              .format(epoch, epoch_time, format_time(end_time - all_start_time)))
        print('AVG train loss {0.avg:.6f}'.format(avg_loss_))

        print("\tLR: {:.4e}".format(lr_scheduler.get_lr()[0]))

        if (epoch + 1) % args.print_freq == 0:
            net.eval()
            accuracy, ave_loss = compute_acc_loss(my_eval, train_loader)
            train_info[epoch + 1] = [ave_loss, accuracy, training_time]
            print('\ttrain loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss, accuracy))

            accuracy, ave_loss = compute_acc_loss(my_eval, test_loader)
            test_info[epoch + 1] = [ave_loss, accuracy, training_time]
            
            print('\ttest  loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss, accuracy))
            if bestacc < accuracy:
                bestacc = accuracy
                to_save = {}
                to_save['args'] = args
                to_save['optimizer_state'] = optimizer.state_dict()
                to_save['model_state'] = net.state_dict()
                to_save['train_info'] = train_info
                to_save['test_info'] = test_info
                to_save['compression_Flops_stats'] = Flops_infor
                to_save['compression_para_stats'] = para_infor
                torch.save(to_save, f'result/{args.arch}_ft_{args.ratio}_epoch{epoch}_{accuracy}.th')
            net.train()

        lr_scheduler.step()

