import os
import matplotlib
import matplotlib.pyplot as plt
from utils.config import Config
from numpy import array
import pandas as pd
import numpy as np
from sklearn import metrics

config = Config()
#config.all_subjects = np.array([1,2,3,4,6,7])
color_codes = ['-g.', '-b.', '-r.']
if config.apply_valid_set:
   datasets = ['train','test','valid']
else:
   datasets = ['test']

#filePath = './out/'

test_acc_major  = {}
valid_acc_major = {}
train_acc_major = {}

'''
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
plt.plot([str(key_val) for key_val in train_acc_major.keys()], list(train_acc_major.values()), '-r.', label='Train (Major Voting)')
plt.plot([str(key_val) for key_val in test_acc_major.keys()],  list(test_acc_major.values()),  '-b.', label='Test (Major Voting)')
if config.apply_valid_set:
   plt.plot([str(key_val) for key_val in valid_acc_major.keys()], list(valid_acc_major.values()), '-g.', label='Valid')

print(f"Test Acc Major: {test_acc_major}")
'''

filePath = './csv_out/'

acc     = {}
precis  = {}
recalls = {}
f1scores= {}
correct = 0
total   = 0
plt.figure(1)
header_string = "subject,acc,precision,recall,f1score"
data_r = np.empty((0,len(header_string.split(','))))

for i, dataset in enumerate(datasets):
    correct = 0
    total   = 0
    for train_idx in config.all_subjects:
        results = pd.read_csv(filePath+dataset+"_"+str(train_idx)+".0.csv", encoding='utf-8')
        precision, recall, f1score, support = metrics.precision_recall_fscore_support(results['label'], results['prediction'], average='binary')
        print(precision, recall, f1score, support)

        correct_pred = results[results['label'] == results['prediction']]
        correct = len(correct_pred)
        total   = len(results['label'])
        acc[train_idx]      = correct/total*100.
        precis[train_idx]   = precision
        recalls[train_idx]  = recall
        f1scores[train_idx] = f1score
        data_r = np.append(data_r, np.array([[train_idx, acc[train_idx], precision, recall, f1score]]), axis = 0)

    plt.plot([str(key_val) for key_val in acc.keys()], list(acc.values()), color_codes[i], label=dataset.capitalize())

if config.realVSFake:
   plt.title("First vs Third")
print(acc.keys())
print([str(key_val) for key_val in acc.keys()])

np.savetxt(filePath+"stats.csv", data_r, delimiter=",", header=header_string, comments='')

print(f"Test Acc: {acc}")



for i, dataset in enumerate(datasets):
    correct = 0
    total   = 0
    for sub in config.all_subjects:
        df = pd.read_csv(filePath+dataset+"_"+str(sub)+".0.csv", encoding='utf-8')
        trials = df['trial'].unique()
        labels = df['label'].unique()
        conditions = df['condition'].unique()
        #print(f"Shapes: T:{len(trials)}, L:{len(labels)}, C:{len(conditions)}")
        correct = 0
        total = 0
        for tr in trials:
            for condition in conditions:
                rows = df[(df['subject'] == sub) & (df['trial'] == tr) & (df['condition'] == condition)]
                correct_pred = rows[rows['label'] == rows['prediction']]
                if len(correct_pred) >= config.threshold*len(rows['label']):
                   correct += 1
                total += 1
        test_acc_major[sub] = 100.*correct/total

plt.plot([str(key_val) for key_val in test_acc_major.keys()],  list(test_acc_major.values()),  '-b.', label='Test (Major Voting)')

plt.xticks([str(key_val) for key_val in acc.keys()])
plt.ylabel("Accuracy (%)")
plt.xlabel("Subject in Test set")
plt.legend()
plt.tight_layout()
plt.savefig('./Figs/Accs.png')
plt.close()

plt.figure(11)
plt.plot([str(key_val) for key_val in acc.keys()], list(acc.values()),          '-g.', label='Accuracy')
plt.plot([str(key_val) for key_val in precis.keys()], list(precis.values()),   '-r.', label='Precision')
plt.plot([str(key_val) for key_val in recalls.keys()], list(recalls.values()),  '-b.', label='Recall')
plt.plot([str(key_val) for key_val in f1scores.keys()], list(f1scores.values()),'-k.', label='F1-Score')
plt.savefig('./Figs/multi_acc.png')
plt.close()

import numpy as np
print(acc)
print(np.mean(list(acc.values())), np.std(list(acc.values())))

print(test_acc_major)
print(np.mean(list(test_acc_major.values())), np.std(list(test_acc_major.values())))

