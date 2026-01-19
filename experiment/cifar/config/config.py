# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser(description='Training SNN')
parser.add_argument('--seed', default=60, type=int, help='random seed')

# model setting
parser.add_argument('--stu_arch', default="resnet18", type=str,
                    help="resnet18|resnet19")
parser.add_argument('--tea_arch', default="resnet34", type=str, help="resnet34")

parser.add_argument('--tea_path', default="your pt file", type=str, help='pth file that holds the model parameters')

# input data preprocess
parser.add_argument('--dataset', default="CIFAR100", type=str, help="CIFAR10|CIFAR100")
parser.add_argument('--data_path', default="your data path", type=str)
parser.add_argument('--log_path', default="./log", type=str, help="log path")
parser.add_argument('--auto_aug', default=True, action='store_true')
parser.add_argument('--cutout', default=True, action='store_true')

# learning setting
parser.add_argument('--optim', default='SGDM', type=str, 
                    help='Optimizer: SGDM (SGD with momentum), ADAM, or ADAMW')
parser.add_argument('--scheduler', default='COSINE', type=str)
parser.add_argument('--train_batch_size', default=128, type=int)
parser.add_argument('--val_batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=5e-4, type=float)
parser.add_argument('--num_epoch', default=300, type=int)
parser.add_argument('--num_workers', default=8, type=int)

# spiking neuron setting
parser.add_argument('--decay', default=0.5, type=float)

parser.add_argument('--thresh', default=1.0, type=float)
parser.add_argument('--T', default=6, type=int, help='num of time steps')
parser.add_argument('--step_mode', default='m', help='step mode')
parser.add_argument('--v_reset', default=0.0, type=float)
parser.add_argument('--detach_reset', default=False, action='store_true')
# training algorithm
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--alpha', default=0.0, type=float, help='weight for Inter-Model Alignment (ELA)')
parser.add_argument('--beta', default=0.0, type=float, help='weight for Intra-SNN Distillation (STA)')
parser.add_argument('--margin', default=0.05, type=float, help='margin for binary entropy loss, target entropy = ln(2) - margin')

# training mode
parser.add_argument('--loss_method', default='seal', type=str, 
                    choices=['seal'],
                    help='Loss method to use: seal (Selective Alignment for SNN)')
parser.add_argument('--baseline', default=False, action='store_true', 
                    help='Deprecated. Use loss_method instead.')

# alignment method for loss_time2
parser.add_argument('--align_method', default='ela', type=str,
                    choices=['ela'],
                    help='Alignment method: ela (Error-aware Logits Alignment)')

# time_step_kd method variant
parser.add_argument('--time_kd_method', default='sta', type=str,
                    choices=['sta'],
                    help='Method for time_step_kd_loss: sta (Similarity-aware Temporal Alignment)')

args = parser.parse_args()
