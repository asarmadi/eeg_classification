import os
import csv
import matplotlib
import matplotlib.pyplot as plt
from utils.config import Config
from numpy import array

config = Config()

filePath = './out/'
#csv_files = os.listdir(filePath)

test_acc  = {}
valid_acc = {}
train_acc = {}

for train_idx in config.all_subjects:
    with open(filePath+"test_results_"+str(train_idx)+".0.csv", newline='') as f:
           data = list(csv.reader(f))
           train_acc[data[2][2][:-2]] = float(data[1][1])
           test_acc[data[2][2][:-2]]  = float(data[2][1])
           valid_acc[data[3][2][:-2]] = float(data[3][1])

print(array([train_acc[k] for k in train_acc]).mean())
print(array([train_acc[k] for k in train_acc]).std())

print(array([test_acc[k] for k in test_acc]).mean())
print(array([test_acc[k] for k in test_acc]).std())

print(array([valid_acc[k] for k in valid_acc]).mean())
print(array([valid_acc[k] for k in valid_acc]).std())


plt.figure(0)
plt.plot(list(train_acc.keys()), list(train_acc.values()), '-r.', label='Train')
plt.plot(list(test_acc.keys()),  list(test_acc.values()),  '-b.', label='Test')
plt.plot(list(valid_acc.keys()), list(valid_acc.values()), '-g.', label='Valid')
plt.legend()
plt.tight_layout()
plt.savefig('./Figs/Accs.png')

