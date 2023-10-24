import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import functional as F

import os
import argparse
from config import *
from model import *
from utils import progress_bar
from hdf5_dataset import *

parser = argparse.ArgumentParser(description='Backdoor Detection')
parser.add_argument('--device', default='cuda:0',type=str, help='GPU device')
parser.add_argument('--batch_size', default=32, type=int, help='Test Batch Size')
parser.add_argument('--num_workers', default=4, type=int, help='Test Batch Size')
args = parser.parse_args()

best_acc = 0
testset  = HDF5Dataset('./data/test_2d.h5')
validset = HDF5Dataset('./data/valid_2d.h5')


testloader  = torch.utils.data.DataLoader(testset,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
validloader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)

inputs, _ = next(iter(testloader))

in_shape=inputs[0,:,:,:].shape
net = Net(in_shape, n_subjects)
net.load_state_dict(torch.load('./checkpoint/Net.pth',map_location=args.device))
net = net.to(args.device)

net.eval()

cudnn.benchmark = True

def test(dataLoader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataLoader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Acc: %.3f%% (%d/%d)'% (100.*correct/total, correct, total))

        print('Test Acc: {0:.3f} ({1}/{2})'.format(100.*correct/total, correct, total))
print("Test Performance:")
test(testloader)
print("Validation Performance:")
test(validloader)




