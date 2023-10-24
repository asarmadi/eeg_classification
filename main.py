import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import argparse
from utils.config import Config
from utils.utils import progress_bar, model_loader, data_loader


parser = argparse.ArgumentParser(description='Backdoor Detection')
parser.add_argument('--device', default='cuda:0',type=str, help='GPU device')
parser.add_argument('--model_type', default='cnn',type=str, help='cnn, caspnet')
parser.add_argument('--batch_size', default=32, type=int, help='Test Batch Size')
parser.add_argument('--n_epochs', default=500, type=int, help='Number of epochs')
parser.add_argument('--num_workers', default=4, type=int, help='Test Batch Size')
parser.add_argument('--lr', default=0.001,type=float,help='Learning Rate')
parser.add_argument('--wd', default=0.01,type=float,help='Weight Decay')
args = parser.parse_args()

best_acc = 0
trainloader, validloader, _ = data_loader(args.batch_size, args.num_workers)
inputs, _ = next(iter(trainloader))
#inputs, _ = next(iter(trainloader))

in_shape=inputs[0,:,:,:].shape
config = Config(args.model_type, in_shape)
net = model_loader(config)
#net.load_state_dict(torch.load('./checkpoint/Net.pth',map_location=args.device))
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
print(inputs.shape, pytorch_total_params)
optimizer = optim.Adam(net.parameters(), lr=args.lr,  weight_decay=args.wd)
#optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr)

cudnn.benchmark = True

#criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()
scheduler = MultiStepLR(optimizer, milestones=[100,200], gamma=0.1)
#scheduler = CyclicLR(optimizer, cosine(t_max=len(trainloader) * 2, eta_min=args.lr/100))

def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
               % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch,args):
    global best_acc
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(validloader), 'Acc: %.3f%% (%d/%d)'% (100.*correct/total, correct, total))

        print('Test Acc: {0:.3f} ({1}/{2})'.format(100.*correct/total, correct, total))
        clean_acc = 100.*correct/total
        if epoch == 1:
           best_acc = clean_acc
        if (clean_acc >= best_acc):
            print('Saving..')
            torch.save(net.state_dict(), './checkpoint/Net.pth')
            best_acc = clean_acc


for epoch in range(1,args.n_epochs):
    print('\nEpoch: {}/{}'.format(epoch,args.n_epochs))
    train(epoch)
    test(epoch,args)

    scheduler.step(epoch)




