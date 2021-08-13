# @Time : 2021/8/3 19:56
# @Author : Deng Xutian
# @Email : dengxutian@126.com

import dataset
import network
import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader


class dl():
    def __init__(self):
        self.lr = 0.001
        self.net = network.env_net()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        self.shuffle = True
        self.num_workers = 1
        self.batch_size = 1
        train_dataset_list = []
        valid_dataset_list = []
        for i in range(1, 10):
            train_dataset_list.append(dataset.env_dataset(object_index=3, file_index=i))
        for i in range(10, 11):
            valid_dataset_list.append(dataset.env_dataset(object_index=3, file_index=i))
        self.train_dataset_loader = DataLoader(ConcatDataset(train_dataset_list), shuffle=self.shuffle,
                                               num_workers=self.num_workers, batch_size=self.batch_size)
        self.valid_dataset_loader = DataLoader(ConcatDataset(valid_dataset_list), shuffle=self.shuffle,
                                               num_workers=self.num_workers, batch_size=self.batch_size)

        self.train_loss_list = []
        self.valid_loss_list = []
        pass

    def save(self):
        torch.save(self.net.state_dict(), 'env_net.pth')
        pass

    def load(self):
        self.net.load_state_dict(torch.load('env_net.pth'))
        pass

    def train(self):
        loss_sum = 0.0
        batch_sum = 0
        loss_func = torch.nn.MSELoss()
        for data in self.train_dataset_loader:
            f, x = data
            x = x.float()
            f = f.float()
            f_ = self.net(x)
            loss = loss_func(f, f_)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            loss_sum = loss_sum + loss.item()
            batch_sum = batch_sum + x.shape[0]
        print('Train Loss = ' + str(loss_sum / batch_sum))
        self.train_loss_list.append(loss_sum / batch_sum)
        pass

    def valid(self):
        loss_sum = 0.0
        batch_sum = 0
        loss_func = torch.nn.MSELoss()
        with torch.no_grad():
            for data in self.valid_dataset_loader:
                f, x = data
                x = x.float()
                f = f.float()
                f_ = self.net(x)
                loss = loss_func(f, f_)

                loss_sum = loss_sum + loss.item()
                batch_sum = batch_sum + x.shape[0]
        print('Valid Loss = ' + str(loss_sum / batch_sum))
        self.valid_loss_list.append(loss_sum / batch_sum)
        pass

    def free_valid(self):
        x = torch.tensor(float(input('Input = '))).float()
        x = torch.unsqueeze(x, dim = 0)
        with torch.no_grad():
            print(self.net(x))

    def save_txt(self):
        train_loss_file = open('train_loss_txt', 'a')
        valid_loss_file = open('valid_loss_txt', 'a')

        try:
            train_loss_str = str(len(self.train_loss_list)) + ' ' + str(
                self.train_loss_list[len(self.train_loss_list) - 1]) + '\n'
        except:
            train_loss_str = '\n'

        try:
            valid_loss_str = str(len(self.valid_loss_list)) + ' ' + str(
                self.valid_loss_list[len(self.valid_loss_list) - 1]) + '\n'
        except:
            valid_loss_str = '\n'

        train_loss_file.write(train_loss_str)
        train_loss_file.close()
        valid_loss_file.write(valid_loss_str)
        valid_loss_file.close()

if __name__ == '__main__':
    demo = dl()
    # demo.load()
    # demo.free_valid()
    for i in range(50):
        demo.train()
        demo.valid()
        demo.save_txt()
    demo.save()
