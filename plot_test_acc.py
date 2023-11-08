import os
import csv
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['xtick.labelsize']=26
matplotlib.rcParams['ytick.labelsize']=26
matplotlib.rcParams['axes.labelsize']=29
matplotlib.rcParams['figure.facecolor']='white'

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 26}

matplotlib.rc('font', **font)
plt.rc('axes', labelsize=26)

filePath = './out/'
csv_files = os.listdir(filePath)
test_acc  = {}
valid_acc = {}
train_acc = {}

for csv_file in csv_files:
    with open(filePath+csv_file, newline='') as f:
           data = list(csv.reader(f))
           train_acc[data[2][2]] = data[1][1]
           test_acc[data[2][2]] = data[2][1]
           valid_acc[data[3][2]] = data[3][1]

plt.figure(0)
plt.plot(list(train_acc.keys()), list(train_acc.values()), '-r.', label='Train')
plt.plot(list(test_acc.keys()), list(test_acc.values()), '-b.', label='Test')
plt.plot(list(valid_acc.keys()), list(valid_acc.values()), '-g.', label='Valid')
plt.legend()
plt.savefig('./Figs/Accs.png')
