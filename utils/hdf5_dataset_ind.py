import h5py
import random
import torch
import os
import sys
sys.path.insert(0, os.path.join(sys.path[0], '..'))

class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, data_type, config, add_trial=False):
        self.data_type       = data_type
        self.add_trial       = add_trial
        self.config          = config

        all_files = os.listdir('./data/separate/')
        self.dataset_len = 0
        for file_name in all_files:
            if self.data_type in file_name:
               self.dataset_len += 1
        print(f'Length of data: {self.dataset_len}')


    def __getitem__(self, index):
        with h5py.File(self.config.file_path+self.data_type+'_2dstfockwell_'+str(index)+'.h5', 'r',rdcc_nbytes=1024**2*4000,rdcc_nslots=1e7) as file:
           if self.add_trial:
              return (file["data"][0],file["label"][0],file["subject"][0],file["trial"][0],file["condition"][0])

           return (file["data"][0],file["label"][0],file["subject"][0])

    def __len__(self):
        return self.dataset_len

