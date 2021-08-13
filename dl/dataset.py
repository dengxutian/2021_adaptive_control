# @Time : 2021/8/3 19:56
# @Author : Deng Xutian
# @Email : dengxutian@126.com

import torch
import random
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, DataLoader


class env_dataset(Dataset):
    def __init__(self, object_index, file_index):
        if object_index == 1:
            dir_path = './ForceObject/arm/'
            pass
        elif object_index == 2:
            dir_path = './ForceObject/bench/'
            pass
        elif object_index == 3:
            dir_path = './ForceObject/wrist/'
            pass
        else:
            print('Object Error!')
        self.data = np.loadtxt(dir_path + str(file_index) + '.txt')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
            return self.data[index, 0], self.data[index, 1] * 1000


if __name__ == '__main__':
    test_dataset = env_dataset(object_index=1, file_index=1)
    dataset_loader = DataLoader(test_dataset, batch_size=10)
    for data in dataset_loader:
        print(data.shape)
