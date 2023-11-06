import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import argparse
import numpy as np
from utils.config import Config
from utils.utils import *


parser = argparse.ArgumentParser(description='Backdoor Detection')
parser.add_argument('--device', default='cuda:0',type=str, help='GPU device')
parser.add_argument('--model_type', default='cnn',type=str, help='cnn, caspnet, lstm')
parser.add_argument('--batch_size', default=32, type=int, help='Test Batch Size')
parser.add_argument('--n_epochs', default=500, type=int, help='Number of epochs')
parser.add_argument('--num_workers', default=4, type=int, help='Test Batch Size')
parser.add_argument('--lr', default=0.001,type=float,help='Learning Rate')
parser.add_argument('--wd', default=0.01,type=float,help='Weight Decay')
args = parser.parse_args()

best_acc = 0
trainloader, validloader, _ = data_loader(args.batch_size, args.num_workers, args.model_type)

config = Config(args.model_type)
net = model_loader(config)
#net.load_state_dict(torch.load('./checkpoint/Net.pth',map_location=args.device))
#net = torch.nn.DataParallel(net)
#net = net.module
net = net.to(args.device)
'''
for m in net.modules():
    if isinstance(m, nn.Conv2d):
#       m.weight.data.normal_(0.0,2/np.sqrt(m.in_channels*m.out_channels*9))
       m.weight.data.normal_(0, 0.05)
       m.bias.data.fill_(0.0)
    if type(m)==nn.Linear:
#       m.weight.data.normal_(0.0,2/np.sqrt(m.in_features))
       torch.nn.init.eye_(m.weight)
       m.bias.data.fill_(0.0)
'''
net.eval()

pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f'Number of Parameters: {pytorch_total_params}')
optimizer = optim.Adam(net.parameters(), lr=args.lr,  weight_decay=args.wd)
#optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr)

cudnn.benchmark = True

criterion = nn.BCELoss()
scheduler = MultiStepLR(optimizer, milestones=[100,200], gamma=0.1)

def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device).reshape(-1,1)
        optimizer.zero_grad()
        if config.model_type == 'capsnet':
           outputs, reconstructions, masked = net(inputs)
           #onehot_tensor = onehot_encode(targets, args.device)
           #loss = net.loss(inputs, outputs, onehot_tensor, reconstructions)
        else:
           outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        predicted = outputs.round()
        total += targets.size(0)
        if config.model_type == 'capsnet':
           correct += sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(onehot_tensor.data.cpu().numpy(), 1))
        else:
           correct += predicted.eq(targets).sum().item()
#           print(predicted, targets, predicted.eq(targets), predicted.eq(targets).sum())
 #          input('enter')

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
               % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

for epoch in range(1,args.n_epochs):
    print('\nEpoch: {}/{}'.format(epoch,args.n_epochs))
    train(epoch)
    clean_acc = test(net, validloader, config.model_type, args.device)
    if epoch == 1:
       best_acc = clean_acc
    if (clean_acc >= best_acc):
       print('Saving..')
       torch.save(net.state_dict(), './checkpoint/Net.pth')
       best_acc = clean_acc


    scheduler.step(epoch)




