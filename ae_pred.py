import torch
import argparse
import torch.backends.cudnn as cudnn
from utils.config import Config
from utils.utils import data_loader,model_loader
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='EEG Classfication Network Testing')
parser.add_argument('--device', default='cuda:0',type=str, help='GPU device')
parser.add_argument('--model_type', default='cnn',type=str, help='cnn, caspnet')
parser.add_argument('--batch_size', default=32, type=int, help='Test Batch Size')
parser.add_argument('--kernel_size', default=32, type=int, help='Model Kernel Size')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers')
parser.add_argument('--transform', default="", type=str, help='Apply transform (e.g., stft, stockwell)')
parser.add_argument('--maj_vote', action='store_true', default=False, help='Majority Voting')
args = parser.parse_args()

best_acc = 0

config = Config(args.model_type)
_, _, testloader = data_loader(args.batch_size, args.num_workers, args.transform,\
                                                   config.apply_valid_set,add_trial=args.maj_vote,config=config)
config.device=args.device
net = model_loader(config, args.kernel_size)
net = net.to(args.device)
#net = torch.nn.DataParallel(net)
net.load_state_dict(torch.load('./checkpoint/Net.pth'))

inputs, targets, _ = next(iter(testloader))

outputs = net(inputs.to(args.device))

fig = plt.figure(1)
axs = fig.subplots(2, 1)
axs[0].plot(inputs[0].cpu().detach().numpy())
axs[0].set_ylabel("Orig")
axs[1].plot(outputs[0].cpu().detach().numpy())
axs[1].set_ylabel("AE")
plt.savefig('./Figs/AE_orig.png')
plt.show()
plt.close()
