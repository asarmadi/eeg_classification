import os
import csv
import matplotlib
import matplotlib.pyplot as plt
from utils.config import Config
from numpy import array
import pandas as pd

config = Config()
config.all_subjects = array([1,2,3,4,6,7,8])
color_codes = ['-g.', '-b.', '-r.']
if config.apply_valid_set:
   datasets = ['train','test','valid']
else:
   datasets = ['test']

filePath = './out/'

test_acc_major  = {}
valid_acc_major = {}
train_acc_major = {}

for train_idx in config.all_subjects:
    with open(filePath+"test_results_"+str(train_idx)+".0.csv", newline='') as f:
           data = list(csv.reader(f))
           train_acc_major[int(data[2][2][1:-2])] = float(data[1][1])
           test_acc_major[int(data[2][2][1:-2])]  = float(data[2][1])
           if config.apply_valid_set:
              valid_acc_major[int(data[2][2][1:-2])] = float(data[3][1])

print(array([train_acc_major[k] for k in train_acc_major]).mean())
print(array([train_acc_major[k] for k in train_acc_major]).std())

print(array([test_acc_major[k] for k in test_acc_major]).mean())
print(array([test_acc_major[k] for k in test_acc_major]).std())

if config.apply_valid_set:
   print(array([valid_acc_major[k] for k in valid_acc_major]).mean())
   print(array([valid_acc_major[k] for k in valid_acc_major]).std())


plt.figure(0)
plt.plot(list(train_acc_major.keys()), list(train_acc_major.values()), '-r.', label='Train (Major Voting)')
plt.plot(list(test_acc_major.keys()),  list(test_acc_major.values()),  '-b.', label='Test (Major Voting)')
if config.apply_valid_set:
   plt.plot(list(valid_acc_major.keys()), list(valid_acc_major.values()), '-g.', label='Valid')
#if config.realVSFake:
#   plt.title("First vs Third")
#plt.legend()
#plt.tight_layout()
#plt.savefig('./Figs/Accs_major.png')


filePath = './csv_out/'

acc     = {}
correct = 0
total   = 0
#plt.figure(1)

for i, dataset in enumerate(datasets):
    correct = 0
    total   = 0
    for train_idx in config.all_subjects:
        results = pd.read_csv(filePath+dataset+"_"+str(train_idx)+".0.csv", encoding='utf-8')
        correct_pred = results[results['label'] == results['prediction']]
        correct += len(correct_pred)
        total   += len(results['label'])
        acc[train_idx] = correct/total*100.
    print(list(acc.keys()))
    plt.plot(list(acc.keys()), list(acc.values()), color_codes[i], label=dataset.capitalize())
if config.realVSFake:
   plt.title("First vs Third")
plt.legend()
plt.tight_layout()
plt.savefig('./Figs/Accs.png')




