import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

from model import ResNet_ANN
from util import Logger, Bar, AverageMeter, accuracy, load_dataset, init_config

def train(train_ldr, optimizer, model, evaluator, args):
    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_ldr))

    for idx, (ptns, labels) in enumerate(train_ldr):
        device = next(model.parameters()).device
        ptns, labels = ptns.to(device), labels.to(device)

        optimizer.zero_grad()
        if len(ptns.shape) == 5:
            # For DVS data, ANN model takes average of frames
            output = model(ptns.mean(1))
        else:
            output = model(ptns)
        loss = evaluator(output, labels)
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(output.data, labels.data, topk=(1, 5))
        losses.update(loss.data.item(), ptns.size(0))
        top1.update(prec1.item(), ptns.size(0))
        top5.update(prec5.item(), ptns.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=idx + 1,
            size=len(train_ldr),
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return top1.avg, losses.avg

def test(val_ldr, model, evaluator, args):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    bar = Bar('Processing', max=len(val_ldr))

    with torch.no_grad():
        for idx, (ptns, labels) in enumerate(val_ldr):
            device = next(model.parameters()).device
            ptns, labels = ptns.to(device), labels.to(device)

            if len(ptns.shape) == 5:
                # For DVS data, ANN model takes average of frames
                output = model(ptns.mean(1))
            else:
                output = model(ptns)
            loss = evaluator(output, labels)

            prec1, prec5 = accuracy(output.data, labels.data, topk=(1, 5))
            losses.update(loss.data.item(), ptns.size(0))
            top1.update(prec1.item(), ptns.size(0))
            top5.update(prec5.item(), ptns.size(0))

            bar.suffix = '({batch}/{size}) Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=idx + 1,
                size=len(val_ldr),
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
        bar.finish()
    return top1.avg, losses.avg

def main():
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log = Logger(args, args.log_path)
    log.info_args(args)
    writer = SummaryWriter(args.log_path)

    # Load Dataset
    if 'dvs' in args.dataset.lower():
        from experiment.dvs.process import butongDVSCifar10
        train_data, val_data = butongDVSCifar10(args.data_path, frames_number=args.T)
        num_class = 10
    else:
        train_data, val_data, num_class = load_dataset(args.dataset, args.data_path, time_step=args.T)

    train_ldr = DataLoader(dataset=train_data, batch_size=args.train_batch_size, shuffle=True,
                           pin_memory=True, num_workers=args.num_workers)
    val_ldr = DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False,
                         pin_memory=True, num_workers=args.num_workers)

    # Model Initialization
    in_channels = 2 if 'dvs' in args.dataset.lower() else 3
    if args.arch in ResNet_ANN.__dict__:
        model = ResNet_ANN.__dict__[args.arch](num_classes=num_class, in_channels=in_channels)
    else:
        raise ValueError(f"Model architecture {args.arch} not supported.")
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch)
    evaluator = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_epoch = 0
    for epoch in range(args.num_epoch):
        train_acc, train_loss = train(train_ldr, optimizer, model, evaluator, args)
        scheduler.step()
        val_acc, val_loss = test(val_ldr, model, evaluator, args)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            state = {
                'best_acc': best_acc,
                'best_epoch': epoch,
                'best_net': model.state_dict(),
            }
            torch.save(state, os.path.join(args.log_path, 'model_weights.pth'))
            log.info('--- Best model saved at epoch %03d with acc %.4f ---' % (epoch, best_acc))

        log.info('Epoch %03d: train_loss %.5f, test_loss %.5f, train_acc %.4f, test_acc %.4f, best_acc %.4f' % (
            epoch, train_loss, val_loss, train_acc, val_acc, best_acc))
        
        writer.add_scalars('Loss', {'val': val_loss, 'train': train_loss}, epoch + 1)
        writer.add_scalars('Acc', {'val': val_acc, 'train': train_acc}, epoch + 1)

    log.info('================================================================')
    log.info('Finish training! Best accuracy: {:.4f} at epoch {}.'.format(best_acc, best_epoch))
    log.info('Best weights saved to: {}'.format(os.path.join(args.log_path, 'model_weights.pth')))
    log.info('================================================================')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Training ANN')
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--log_path', default='./log/ann', type=str)
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--val_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--wd', default=5e-4, type=float)
    parser.add_argument('--num_epoch', default=300, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--seed', default=60, type=int)
    parser.add_argument('--save_features', action='store_true') # Added to support user command
    parser.add_argument('--T', default=10, type=int) # Added to support user command
    parser.add_argument('--optim', default='SGDM', type=str) # Added for compatibility
    parser.add_argument('--stu_arch', default='none', type=str) # Added for get_model_name compatibility
    parser.add_argument('--cutout', default=False, action='store_true')
    parser.add_argument('--auto_aug', default=False, action='store_true')
    args = parser.parse_args()
    
    # Use stu_arch as arch for naming consistency in init_config
    if args.stu_arch == 'none':
        args.stu_arch = args.arch

    init_config(args)
    main()
