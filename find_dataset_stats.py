import argparse
from utils.utils import *
from utils.config import Config

parser = argparse.ArgumentParser(description='Backdoor Detection')
parser.add_argument('--device', default='cuda:0',type=str, help='GPU device')
parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers')
args = parser.parse_args()

config = Config('','')

trainloader, validloader, testloader = data_loader(args.batch_size, args.num_workers)

for condition in config.conditions:
    count_num_classes(testloader, condition)