import torch
import argparse
import torch.backends.cudnn as cudnn
from utils.config import Config
from utils.utils import *

parser = argparse.ArgumentParser(description='Backdoor Detection')
parser.add_argument('--device', default='cuda:0',type=str, help='GPU device')
parser.add_argument('--model_type', default='cnn',type=str, help='cnn, caspnet')
parser.add_argument('--batch_size', default=32, type=int, help='Test Batch Size')
parser.add_argument('--num_workers', default=4, type=int, help='Test Batch Size')
args = parser.parse_args()

best_acc = 0
trainloader, validloader, testloader = data_loader(args.batch_size, args.num_workers)

config = Config(args.model_type)
net = model_loader(config)
net.load_state_dict(torch.load('./checkpoint/Net.pth',map_location=args.device))
net = net.to(args.device)

net.eval()

cudnn.benchmark = True

print("Test Performance:")
test(net, testloader, config.model_type, args.device)
print("Validation Performance:")
test(net, validloader, config.model_type, args.device)




