import h5py
from utils.config import Config
from utils.utils import preprocess_signal
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

path = config.file_path + "SF_Img_MLdata.mat"

def generate_data(data_type):
    u = 0
    if data_type == 'train':
        subjects_list = config.train_subjects
    elif data_type == 'test':
        subjects_list = config.test_subjects
    elif data_type == 'valid':
        subjects_list = config.valid_subjects
    n_samples = len(subjects_list)*config.n_windows*config.n_trial*len(config.conditions)
    data_shape = (n_samples, config.window_len, config.n_channels)
    f_data = h5py.File(config.file_path+data_type+'_1d.h5', "w")
    f_data.create_dataset("data",    data_shape)
    f_data.create_dataset("label",   (n_samples,))
    f_data.create_dataset("subject", (n_samples,))
    f_data.create_dataset("trial",   (n_samples,))

    f = h5py.File(path,'r')

    for subject in subjects_list:
        for condition in config.conditions: # 0, 1 correpond to Flex First, and Extend First
            print(f'{data_type} subject#: {subject}, condition: {condition}')
            ref = f["data"][condition][subject]
            eeg = np.array(f[ref])
            for i_trial in range(config.n_trial):
                eeg_scaled = eeg[i_trial,:,:]
                eeg_scaled = preprocess_signal(config,eeg_scaled)
                for j_windows in range(config.n_windows):
                    eeg_norm = eeg_scaled[j_windows*config.window_inc:j_windows*config.window_inc+config.window_len,:]
                    f_data["data"][u,...] = eeg_norm
                    f_data["subject"][u]  = subject
                    f_data["label"][u]    = condition
                    f_data["trial"][u]    = i_trial
                    u += 1
    f_data.close()

for dataSet in ['train', 'test', 'valid']:
    generate_data(dataSet)

