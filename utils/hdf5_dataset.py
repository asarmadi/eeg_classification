import h5py
import random
import torch

class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.file_path       = path
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file["data"])

    def __getitem__(self, index):
        data = h5py.File(self.file_path, 'r',rdcc_nbytes=1024**2*4000,rdcc_nslots=1e7)
        self.dataset = data["data"]
        self.label   = data["label"]

        x = self.dataset[index]
        return (x,self.label[index])

    def __len__(self):
        return self.dataset_len
