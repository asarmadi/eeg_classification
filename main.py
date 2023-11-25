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
parser.add_argument('--num_workers', default=4, type=int, help='Test Batch Size')
parser.add_argument('--lr', default=0.001,type=float,help='Learning Rate')
parser.add_argument('--wd', default=0.01,type=float,help='Weight Decay')
parser.add_argument('--stft', action='store_true', default=False, help='Apply STFT')
args = parser.parse_args()

best_acc = 0
trainloader, validloader, _ = data_loader(args.batch_size, args.num_workers, args.stft)
config = Config(args.model_type)
config.device = args.device
config.stft=args.stft
gauss_obj = GaussianFourierFeatureTransform(1, config.mapping_size, 10)
net = model_loader(config,args.kernel_size)
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
optimizer = optim.AdamW(net.parameters(), lr=args.lr,  weight_decay=args.wd)
#optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr)

cudnn.benchmark = True

#criterion = nn.BCELoss()
criterion = nn.NLLLoss()
scheduler = MultiStepLR(optimizer, milestones=[200,300], gamma=0.1)

def train():
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, _) in enumerate(trainloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        if config.apply_gauss:
           inputs = gauss_obj(inputs.reshape(-1,1,config.window_len,config.n_channels))
#        if config.model_type == 'eegnet':
 #          inputs = inputs.permute(0,2,1)
#           inputs = inputs.unsqueeze(1)
 #          inputs = inputs.reshape(-1,1,config.window_len,config.n_channels)
        optimizer.zero_grad()
        if config.model_type == 'capsnet':
           outputs, reconstructions, masked = net(inputs)
#           onehot_tensor = onehot_encode(targets, args.device)
#           loss = net.loss(inputs, outputs, targets, reconstructions)
           outputs = outputs.reshape(-1,1)
           outputs = nn.functional.softmax(outputs,dim=1)
        else:
           outputs = net(inputs)
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
#        predicted = outputs.round()
        _, predicted = outputs.max(1)
        total += targets.size(0)
#        if config.model_type == 'capsnet':
 #          correct += sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(onehot_tensor.data.cpu().numpy(), 1))
  #      else:
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
               % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

for epoch in range(1,args.n_epochs):
    print('\nEpoch: {}/{}'.format(epoch,args.n_epochs))
    train()

    clean_acc,_ = test(net, validloader, config)
    if epoch == 1:
       best_acc = clean_acc
    if (clean_acc >= best_acc):
#    if epoch%10==0:
       print('Saving..')
       torch.save(net.state_dict(), './checkpoint/Net.pth')
       best_acc = clean_acc


    scheduler.step(epoch)




