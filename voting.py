import torch
import argparse
import torch.backends.cudnn as cudnn
from utils.config import Config
from utils.utils import *
from utils.hdf5_dataset import HDF5Dataset

parser = argparse.ArgumentParser(description='EEG Classfication Network Testing')
parser.add_argument('--device', default='cuda:0',type=str, help='GPU device')
parser.add_argument('--model_type', default='cnn',type=str, help='cnn, caspnet')
parser.add_argument('--batch_size', default=32, type=int, help='Test Batch Size')
parser.add_argument('--kernel_size', default=32, type=int, help='Model Kernel Size')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers')
parser.add_argument('--transform', default="nothing", type=str, help='Apply transform (e.g., stft, stockwell)')
parser.add_argument('--maj_vote', action='store_true', default=False, help='Majority Voting')
parser.add_argument('--target_test', default='0',type=str, help='Subject for test')
args = parser.parse_args()

best_acc = 0

config = Config(args.model_type)
config.test_subjects  = np.array([int(args.target_test)])
config.train_subjects = np.setdiff1d(config.all_subjects, config.test_subjects)
config.device=args.device
config.maj_vote = args.maj_vote
config.transform = args.transform
cudnn.benchmark = True
net_list = []
for subject in config.train_subjects:
    net = model_loader(config, args.kernel_size)
    net = net.to(args.device)
    net.load_state_dict(torch.load('./checkpoint/Net_'+config.model_type+'_'+str(subject)+'.pth'))
    net.eval()
    net_list.append(net)

testset  = HDF5Dataset('./data/train_1d_'+args.target_test+'.h5', config)
dataloader  = torch.utils.data.DataLoader(testset,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)


trans_net = None
if config.transform != "nothing":
   trans_net = trans_loader(config)
   tras_net.eval()

correct = 0
total = 0
with torch.no_grad():
     for batch_idx, (inputs, targets, subjects) in enumerate(dataloader):
         inputs = inputs.to(config.device)
         results = np.zeros((len(net_list),inputs.shape[0]))
         for i, net_i in enumerate(net_list):
             outputs = get_outputs(net_i, inputs, config, transformer=trans_net)
             _, predicted = outputs.max(1)
             results[i,:] = predicted.cpu().numpy()
         pred = np.mean(results, axis=0)
         pred = np.where(pred>0.5, 1, 0)
         print(results)
         print(pred)
         print(targets.numpy())
         total += targets.size(0)
         correct += (pred == targets.numpy()).sum()
         print(f"{correct}/{total} = {100*correct/total}")
         input('enter')
         progress_bar(batch_idx, len(dataloader), 'Acc: %.3f%% (%d/%d)'% (100.*correct/total, correct, total))

print(f"Test Accuracy: {correct}/{total} = {100*correct/total}")
