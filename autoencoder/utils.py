import torch
import numpy as np

from torch.utils.data import Dataset


def split_dataset(ds):
    T = ds.shape[0]
    t = T // 3
    train = ds[:t, :]
    valid = ds[t:2*t, :]
    tests = ds[2*t:, :]
    return train, valid, tests


def torch_rolling_mean(factors, device):
    t = factors.shape[0]
    divider = torch.tensor(np.arange(1, t+1)).view(t, 1, 1).to(device)
    return factors.cumsum(dim=0).divide(divider)[1:]


class AEDataset(Dataset):

    def __init__(self, char, port, rets):

        self.char = torch.from_numpy(char).float()
        self.port = torch.from_numpy(port).float()
        self.rets = torch.from_numpy(rets).float()
        self.size = self.char.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.char[idx], self.port[idx], self.rets[idx]
