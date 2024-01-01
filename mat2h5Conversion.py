import h5py
from utils.config import Config
from utils.utils import preprocess_signal, spectrogram_per_channel, stockwell
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='EEG h5 generator')
parser.add_argument('--target_test', default='0',type=str, help='Subject for test')
parser.add_argument('--apply_transform', default='', type=str, help='Apply transformatio (e.g., stft, stockwell)')
args = parser.parse_args()

config = Config()
config.test_subjects  = np.array([int(args.target_test)])
config.train_subjects = np.setdiff1d(config.all_subjects, config.test_subjects)
all_trials   =  np.array(range(0,config.n_trial))
test_trials  = all_trials

if config.apply_valid_set:
   valid_sub = np.random.choice(config.train_subjects,3,replace=False)
   config.valid_subjects = np.array(valid_sub)
   config.train_subjects = np.setdiff1d(config.train_subjects, config.valid_subjects)
   valid_trials = all_trials
   print(f'Valid Subjects: {config.valid_subjects}')
   data_sets = ['train', 'test', 'valid']
else:
   data_sets = ['train', 'test']

print(f'Test Subjects:  {config.test_subjects}')
print(f'Train Subjects: {config.train_subjects}')

path = config.file_path + "SF_"+config.data_path+"_MLdata.mat"

def find_subjects_list(data_type):
    if data_type == 'train':
        subjects_list = config.train_subjects
    elif data_type == 'test':
        subjects_list = config.test_subjects
    elif data_type == 'valid':
        subjects_list = config.valid_subjects
    return subjects_list

def find_num_samples(num_trials, sub_list):
    n_samples = 0
    for sub in sub_list:
        for cond in config.conditions:
            n_samples += config.n_windows*num_trials[sub][cond]
    return n_samples

def find_num_trials(data_type):
    data_dict = {}
    f = h5py.File(path,'r')
    subjects_list = find_subjects_list(data_type)
    for subject in subjects_list:
        trials_shape = []
        for condition in config.conditions:
            ref = f["data"][condition][subject]
            eeg = np.array(f[ref])
            trials_shape.append(eeg.shape[0])
        data_dict[subject]=trials_shape
    return data_dict


def generate_data(data_type):
    u = 0
    subjects_list = find_subjects_list(data_type)
    trials_list   = find_num_trials(data_type)

    n_samples = find_num_samples(trials_list,subjects_list)
    print(n_samples)
    if args.apply_transform == 'stft':
        data_shape = (n_samples, config.n_channels, config.freq_cut, config.nTimeBins)
        chunk_shape = (1, config.n_channels, config.freq_cut, config.nTimeBins)
        path_name_str = '2dstft'
    elif args.apply_transform == 'stockwell':
        fmax_samples = int(config.fmax*config.sig_time)
        data_shape = (n_samples, config.n_channels, fmax_samples+1, config.window_len)
        chunk_shape = (1, config.n_channels, fmax_samples+1, config.window_len)
        print(data_shape)
        path_name_str = '2dstfockwell'
    else:
        data_shape = (n_samples, config.n_channels, config.window_len)
        chunk_shape = (1, config.n_channels, config.window_len)
        path_name_str = '1d'
    compression_type = None
    chunks_type = (10,)
    f_data = h5py.File(config.file_path+data_type+'_'+path_name_str+'.h5', "w")
    f_data.create_dataset("data",      data_shape,    compression=compression_type)
    f_data.create_dataset("label",     (n_samples,),  compression=compression_type)
    f_data.create_dataset("subject",   (n_samples,),  compression=compression_type)
    f_data.create_dataset("trial",     (n_samples,),  compression=compression_type)
    f_data.create_dataset("condition", (n_samples,),  compression=compression_type)
#    d1, d2, d3, d4, d5 = np.memmap("data.mymemmap",dtype='float32', mode='w+', shape=data_shape), np.memmap("label.memmap",dtype='float32', mode='w+', shape=(n_samples,)), np.memmap("subject.memamp",dtype='float32', mode='w+', shape=(n_samples,)), np.memmap("trial.memmap",dtype='float32', mode='w+', shape=(n_samples,)),np.memmap("asddas.memmap",dtype='float32', mode='w+', shape=(n_samples,))

    f = h5py.File(path,'r',libver='latest')

    for subject in subjects_list:
        for condition in config.conditions:
            ref = f["data"][condition][subject]
            eeg = np.array(f[ref])
            print(f'{data_type} subject#: {subject}, condition: {condition}')
            ref = f["data"][condition][subject]
            eeg = np.array(f[ref])
            for i_trial in range(eeg.shape[0]):
                eeg_scaled = eeg[i_trial,:,:]
                eeg_scaled = preprocess_signal(config,eeg_scaled)
                for j_windows in range(config.n_windows):
                    eeg_norm = eeg_scaled[j_windows*config.window_inc:j_windows*config.window_inc+config.window_len,:]
                    if args.apply_transform == 'stft':
#                       d1[u,:] = spectrogram_per_channel(eeg_norm, config)
                        f_data["data"][u,...] = spectrogram_per_channel(eeg_norm, config)
                    elif args.apply_transform == 'stockwell':
#                       d1[u,:] = stockwell(eeg_norm, config)
                        f_data["data"][u,...] = stockwell(eeg_norm, config)
                    else:
#                       d1[u,:] = eeg_norm.T
                        f_data["data"][u,...] = eeg_norm.T
                    if config.realVSFake:
                       if condition == 0 or condition == 1:
#                          d2[u] = 0
                          f_data["label"][u]    = 0
                       else:
#                          d2[u] = 1
                          f_data["label"][u]    = 1
                    else:
#                       d2[u] = condition - 2
                       f_data["label"][u]    = condition - 2
                    f_data["trial"][u]    = i_trial
                    f_data["condition"][u]    = condition
                    f_data["subject"][u]  = subject
#                    d3[u] = subject
 #                   d4[u] = i_trial
  #                  d5[u] = condition
                    u += 1

#    f_data = h5py.File(config.file_path+data_type+'_'+path_name_str+'.h5', "w")
 #   f_data.create_dataset("data",      data_shape,    compression=compression_type, data=d1)
  #  f_data.create_dataset("label",     (n_samples,),  compression=compression_type, data=d2)
   # f_data.create_dataset("subject",   (n_samples,),  compression=compression_type, data=d3)
    #f_data.create_dataset("trial",     (n_samples,),  compression=compression_type, data=d4)
    #f_data.create_dataset("condition", (n_samples,),  compression=compression_type, data=d5)

    f_data.close()
#    del d1
#    del d2
#    del d3
#    del d4
#    del d5

for dataSet in data_sets:
    generate_data(dataSet)

