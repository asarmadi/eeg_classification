import h5py
import random
import torch
#import os

class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, add_trial=False):
        self.file_path       = path
        self.add_trial       = add_trial

        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file["data"])

    def __getitem__(self, index):
        data = h5py.File(self.file_path, 'r',rdcc_nbytes=1024**2*4000,rdcc_nslots=1e7)
        self.dataset = data["data"]
        self.label   = data["label"]
        self.subject = data["subject"]
        if self.add_trial:

           return (self.dataset[index],self.label[index],self.subject[index],data["trial"][index],data["condition"][index])

        return (self.dataset[index],self.label[index],self.subject[index])

    def __len__(self):
        return self.dataset_len

