import torch
import argparse
import torch.backends.cudnn as cudnn
from utils.config import Config
from utils.utils import *

parser = argparse.ArgumentParser(description='EEG Classfication Network Testing')
parser.add_argument('--device', default='cuda:0',type=str, help='GPU device')
parser.add_argument('--model_type', default='cnn',type=str, help='cnn, caspnet')
parser.add_argument('--batch_size', default=32, type=int, help='Test Batch Size')
parser.add_argument('--kernel_size', default=32, type=int, help='Model Kernel Size')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers')
parser.add_argument('--stft', action='store_true', default=False, help='Apply STFT')
parser.add_argument('--maj_vote', action='store_true', default=False, help='Majority Voting')
args = parser.parse_args()

best_acc = 0

config = Config(args.model_type)
trainloader, validloader, testloader = data_loader(args.batch_size, args.num_workers, args.stft, config.apply_valid_set,args.maj_vote)
config.device=args.device
config.maj_vote = args.maj_vote
net = model_loader(config, args.kernel_size)
net = net.to(args.device)
#net = torch.nn.DataParallel(net)
net.load_state_dict(torch.load('./checkpoint/Net.pth'))

#net = net.to(args.device)

net.eval()

cudnn.benchmark = True

print("Test Performance:")
test_acc, target_test  = test(net, testloader, config, "test")

if config.apply_valid_set:
   print("Validation Performance:")
   valid_acc, target_valid = test(net, validloader, config, "valid")

print("Train Performance:")
train_acc, _ = test(net, trainloader, config, "train")

header_name = 'Name,Acc,Target'
if config.apply_valid_set:
   data = [['Train', train_acc, 0], ['Test', test_acc, target_test], ['Valid', valid_acc, target_valid] ]
else:
   data = [['Train', train_acc, 0], ['Test', test_acc, target_test] ]

np.savetxt("./out/test_results_"+str(target_test[0])+".csv", data, delimiter=",", header=header_name, comments='', fmt="%s")



