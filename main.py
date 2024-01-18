import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import argparse
import numpy as np
from utils.config import Config
from utils.utils import *
from utils.gaussianTrans import GaussianFourierFeatureTransform


parser = argparse.ArgumentParser(description='EEG Classification')
parser.add_argument('--device', default='cuda:0',type=str, help='GPU device')
parser.add_argument('--model_type', default='cnn',type=str, help='cnn, caspnet, lstm')
parser.add_argument('--batch_size', default=32, type=int, help='Test Batch Size')
parser.add_argument('--kernel_size', default=32, type=int, help='Model Kernel Size')
parser.add_argument('--n_epochs', default=500, type=int, help='Number of epochs')
parser.add_argument('--num_workers', default=8, type=int, help='Test Batch Size')
parser.add_argument('--lr', default=0.001,type=float,help='Learning Rate')
parser.add_argument('--wd', default=0.01,type=float,help='Weight Decay')
parser.add_argument('--transform', default="nothing", type=str, help='Apply transform (e.g., stft, stockwell)')
args = parser.parse_args()

best_acc = 0
config = Config(args.model_type)
config.device = args.device
config.transform=args.transform

if not os.path.isdir('./checkpoint/'):
   os.makedirs('./checkpoint/')

trainloader, validloader, _ = data_loader(args.batch_size, args.num_workers, args.transform, config.apply_valid_set,config=config)

net = model_loader(config,args.kernel_size)
#net= nn.DataParallel(net)
net = net.to(args.device)
#net.load_state_dict(torch.load('./checkpoint/Net_'+config.model_type+'.pth'))
net.eval()

trans_net = None
if config.transform != "nothing":
   trans_net = trans_loader(config)

pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f'Number of Parameters: {pytorch_total_params}')
optimizer = optim.AdamW(net.parameters(), lr=args.lr,  weight_decay=args.wd)

cudnn.benchmark = True

#criterion = nn.BCELoss()
if 'eeg' in config.model_type:
   criterion = torch.nn.CrossEntropyLoss()
elif 'ae' in config.model_type:
   criterion = torch.nn.MSELoss()
else:
   criterion = nn.NLLLoss()
scheduler = MultiStepLR(optimizer, milestones=[200,300], gamma=0.1)

def train():
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, _) in enumerate(trainloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()
        outputs = get_outputs(net, inputs,config, transformer=trans_net)
        if 'ae' in config.model_type:
           loss = criterion(outputs, inputs)
        else:
           loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        total += targets.size(0)
        if not 'ae' in config.model_type:
           _, predicted = outputs.max(1)
           correct += predicted.eq(targets).sum().item()
           progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        else:
           progress_bar(batch_idx, len(trainloader), 'Loss: %.3f' % (train_loss/(batch_idx+1)))

    print(f'Total: {total}')
    return (correct/total)

for epoch in range(1,args.n_epochs):
    print('\nEpoch: {}/{}'.format(epoch,args.n_epochs))
    clean_acc = train()

    if config.apply_valid_set:
       clean_acc,_ = test(net, validloader, config, "valid")
    if epoch == 1:
       best_acc = clean_acc
    if (clean_acc >= best_acc):
       print('Saving..')
       torch.save(net.state_dict(), './checkpoint/Net_'+config.model_type+'.pth')
       best_acc = clean_acc


    scheduler.step(epoch)

