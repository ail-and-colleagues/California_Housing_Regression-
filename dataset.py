import os
import numpy as np
# import pandas as pd
import torch
from torch.utils.data import Dataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Dataset_Template(Dataset):
    def __init__(self, data_path, batch_size, batch_p_ep):
        self.batch_size = batch_size
        self.batch_p_ep = batch_p_ep
        # implementation
        self.data = np.loadtxt(data_path, delimiter=',', skiprows=1)
        print('self.data[:3]: ', self.data[:3])
        self.data = self.data.astype(np.float32)

        # The data should be scaled around -1.0 to 1.0.
        # Min-max normalization or z-score normalization is generally adopted at the loading process.
        mx = np.max(self.data[:, 1:], axis=0)
        mn = np.min(self.data[:, 1:], axis=0)
        self.data[:, 1:] = (mx - mn) * (self.data[:, 1:] - mn)
        self.data[:, 0] = self.data[:, 0] / 1000000.0

        print('normalized')
        print('self.data[:3]: ', self.data[:3])
        print('self.data.shape: ', self.data.shape)

    def __len__(self):
        # __len__ is designed for defining the length of data trained in one epoch.
        # Thus __len__ generally returns the length of data used for training, i.e., "self.data.shape[0]" is returned.
        # On the other hand, data augmentation can be employed in __getitem_, it makes the data counts vague. 
        # In this dataset, __len__ is implemented to return "batch_size *  batch_per_epoch" for easy adjustment.
        return self.batch_size * self.batch_p_ep
    
    def __getitem__(self, index):
        # Argument "index" is ignored because this dataset returns an item regardless of the length of the data.
        
        # implementation
        rid = np.random.choice(self.data.shape[0], 1, replace=False)
        x = self.data[rid, 1:]
        y = self.data[rid, 0]
        x = x.reshape([-1])
        return x, y

if __name__ == '__main__':
    # chk
    batch_size = 10
    batch_p_ep = 100
    train_dataset = Dataset_Template('./datasets/cadata.txt', batch_size, batch_p_ep)

    x, y = train_dataset.__getitem__(0)
    print('x.shape:', x.shape)
    print(x)
    print('y.shape:', y.shape)
    print(y)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    x, y = next(iter(train_loader))
    print('x.shape:', x.shape)
    print(x)
    print('y.shape:', y.shape)
    print(y)