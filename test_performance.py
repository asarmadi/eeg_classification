import torch
import argparse
import torch.backends.cudnn as cudnn
from utils.config import Config
from utils.utils import progress_bar, model_loader, data_loader

parser = argparse.ArgumentParser(description='Backdoor Detection')
parser.add_argument('--device', default='cuda:0',type=str, help='GPU device')
parser.add_argument('--model_type', default='cnn',type=str, help='cnn, caspnet')
parser.add_argument('--batch_size', default=32, type=int, help='Test Batch Size')
parser.add_argument('--num_workers', default=4, type=int, help='Test Batch Size')
args = parser.parse_args()

best_acc = 0
trainloader, validloader, testloader = data_loader(args.batch_size, args.num_workers)
inputs, _ = next(iter(testloader))
in_shape=inputs[0,:,:,:].shape

config = Config(args.model_type, in_shape)
net = model_loader(config)
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




