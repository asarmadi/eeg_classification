import h5py
from utils.config import Config
from utils.utils import Normalize
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser(description='EEG h5 generator')
parser.add_argument('--target_test', default='0',type=str, help='Subject for test')
args = parser.parse_args()

config = Config()
#test_sub = random.choice(config.all_subjects)
config.test_subjects  = np.array([int(args.target_test)])
config.train_subjects = np.setdiff1d(config.all_subjects, config.test_subjects)

valid_sub = np.random.choice(config.train_subjects,3,replace=False)
config.valid_subjects = np.array(valid_sub)
config.train_subjects = np.setdiff1d(config.train_subjects, config.valid_subjects)

all_trials   =  np.array(range(0,config.n_trial))
test_trials  = all_trials
valid_trials = all_trials
#test_trials  = random.choices(all_trials,k=len(all_trials)//2)
#valid_trials = np.setdiff1d(all_trials, test_trials)

print(f'Test Subjects:  {config.test_subjects}')
print(f'Valid Subjects: {config.valid_subjects}')
print(f'Train Subjects: {config.train_subjects}')

n_samples_train = len(config.train_subjects)*config.n_windows*config.n_trial*len(config.conditions)
n_samples_valid = len(config.valid_subjects)*config.n_windows*len(valid_trials)*len(config.conditions)
n_samples_test  = len(config.test_subjects) *config.n_windows*len(test_trials)*len(config.conditions)

print(n_samples_train)

train_shape = (n_samples_train, config.window_len, config.n_channels)
valid_shape = (n_samples_valid, config.window_len, config.n_channels)
test_shape  = (n_samples_test, config.window_len,  config.n_channels)

f_train = h5py.File(config.file_path+'train_1d.h5', "w")
f_train.create_dataset("data",    train_shape)
f_train.create_dataset("label",   (n_samples_train,))
f_train.create_dataset("subject", (n_samples_train,))
f_train.create_dataset("trial",   (n_samples_train,))

f_test = h5py.File(config.file_path+'test_1d.h5', "w")
f_test.create_dataset("data",    test_shape)
f_test.create_dataset("label",   (n_samples_test,))
f_test.create_dataset("subject", (n_samples_test,))
f_test.create_dataset("trial",   (n_samples_test,))


f_valid = h5py.File(config.file_path+'valid_1d.h5', "w")
f_valid.create_dataset("data",    valid_shape)
f_valid.create_dataset("label",   (n_samples_valid,))
f_valid.create_dataset("subject", (n_samples_valid,))
f_valid.create_dataset("trial",   (n_samples_valid,))

u = 0
s = 0    # sth sample in test set
v = 0    # vth sample in validation set

path = config.file_path + "SF_Obs_MLdata.mat"

f = h5py.File(path,'r')

if config.preprocess_normalize:
   train_eeg = []
   for subject in config.train_subjects:
       for condition in config.conditions: # 0, 1 correpond to Flex First, and Extend First
           print(f'subject#: {subject}, condition: {condition}')
           ref = f["data"][condition][subject]
           eeg = np.array(f[ref])
           train_eeg.append(eeg[:,:config.window_len,::])
#
   train_eeg = np.array(train_eeg)
   train_eeg = np.mean(train_eeg, axis=0)
   mean, std = np.mean(train_eeg, axis=0), np.std(train_eeg, axis=0)

for subject in config.train_subjects:
    for condition in config.conditions: # 0, 1 correpond to Flex First, and Extend First
        print(f'Train subject#: {subject}, condition: {condition}')
        ref = f["data"][condition][subject]
        eeg = np.array(f[ref])
        for i_trial in range(config.n_trial):
            for j_windows in range(config.n_windows):
                eeg_norm = eeg[i_trial,j_windows*config.window_inc:j_windows*config.window_inc+config.window_len,:]
                if config.preprocess_normalize:
                   eeg_norm = Normalize(mean, std, eeg_norm)
                f_train["data"][u,...] = eeg_norm
                f_train["subject"][u]  = subject
                f_train["label"][u]    = condition
                f_train["trial"][u]    = i_trial
                u += 1

for subject in config.test_subjects:
    for condition in config.conditions: # 0, 1 correpond to Flex First, and Extend First
        print(f'Test subject#: {subject}, condition: {condition}')
        ref = f["data"][condition][subject]
        eeg = np.array(f[ref])
        for i_trial in test_trials:
            for j_windows in range(config.n_windows):
                eeg_norm = eeg[i_trial,j_windows*config.window_inc:j_windows*config.window_inc+config.window_len,:]
                if config.preprocess_normalize:
                   eeg_norm = Normalize(mean, std, eeg_norm)
                f_test["data"][s,...] = eeg_norm
                f_test["subject"][s]  = subject
                f_test["label"][s]    = condition
                f_test["trial"][s]    = i_trial
                s += 1

for subject in config.valid_subjects:
    for condition in config.conditions: # 0, 1 correpond to Flex First, and Extend First
        print(f'Valid subject#: {subject}, condition: {condition}')
        ref = f["data"][condition][subject]
        eeg = np.array(f[ref])
        for i_trial in valid_trials:
            for j_windows in range(config.n_windows):
                eeg_norm = eeg[i_trial,j_windows*config.window_inc:j_windows*config.window_inc+config.window_len,:]
                if config.preprocess_normalize:
                   eeg_norm = Normalize(mean, std, eeg_norm)
                f_valid["data"][v,...] = eeg_norm
                f_valid["subject"][v]  = subject
                f_valid["label"][v]    = condition
                f_valid["trial"][v]    = i_trial
                v += 1

f_train.close()
f_test.close()
f_valid.close()

