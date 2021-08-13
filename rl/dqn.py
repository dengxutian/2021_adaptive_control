# @Time : 2021/8/3 10:04
# @Author : Deng Xutian
# @Email : dengxutian@126.com


import env
import network
import math
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

import sys

sys.path.append('../')
from dl.network import env_net


class dqn():
    def __init__(self):
        self.x = 0
        self.k = 0
        self.stride = 0.01
        self.f_target = 10
        self.noise = 0

        self.eval_net = network.q_net()
        self.target_net = network.q_net()
        self.env_net = env_net()

        for target_param, eval_param in zip(self.target_net.parameters(), self.eval_net.parameters()):
            target_param.data.copy_(eval_param)

        self.lr = 0.01
        self.optim = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

        self.buffer = []
        self.buffer_len = 1000

        self.soft_update_rate = 0.01

        self.reward_list = []
        self.loss_list = []

    def save(self):
        torch.save(self.target_net.state_dict(), 'q_net.pth')

    def load(self):
        self.target_net.load_state_dict(torch.load('q_net.pth'))
        self.eval_net.load_state_dict(torch.load('q_net.pth'))

    def load_env_net(self):
        self.env_net.load_state_dict(torch.load('env_net.pth'))

    def pre_explore(self):
        x = self.x
        k = self.k
        stride = self.stride
        f_target = self.f_target
        for i in range(self.buffer_len):
            f = env.elasticity(x) + (0.5 - random.random()) * self.noise
            e = f - f_target
            if random.random() > 0.2:
                k = k
                a = 2
            else:
                if random.random() > 0.5:
                    k = k + stride
                    a = 1
                else:
                    k = k - stride
                    a = 0
            x_ = env.pid(x, k, e)
            f_ = env.elasticity(x_) + (0.5 - random.random()) * self.noise
            e_ = f_ - f_target
            r = -math.fabs(e_)
            if a == 2:
                r = r + 10
            self.buffer.append([x, f, e, a, x_, f_, e_, r])
            x = x_

    def explore(self, greedy_rate):
        x = self.x
        k = self.k
        stride = self.stride
        f_target = self.f_target
        reward = 0
        for i in range(100):
            f = env.elasticity(x) + random.random() * self.noise
            e = f - f_target
            if random.random() > greedy_rate:
                with torch.no_grad():
                    action = self.eval_net(torch.from_numpy(np.array([x, f, e])).float())
                    action = torch.argmax(action)
                    a = action.item()
                    if a == 0:
                        k = k - stride
                    if a == 1:
                        k = k + stride
                    if a == 2:
                        k = k
            else:
                if random.random() > 0.2:
                    k = k
                    a = 2
                else:
                    if random.random() > 0.5:
                        k = k + stride
                        a = 1
                    else:
                        k = k - stride
                        a = 0
            x_ = env.pid(x, k, e)
            f_ = env.elasticity(x_) + (0.5 - random.random()) * self.noise
            e_ = f_ - f_target
            r = -math.fabs(e_)
            if a == 2:
                r = r + 10
            reward = reward + r
            if len(self.buffer) > self.buffer_len:
                self.buffer.pop(0)
            self.buffer.append([x, f, e, a, x_, f_, e_, reward])  # state, action, state_, reward
            x = x_
        self.reward_list.append(reward)

    def train(self):
        gamma = 0.9
        loss_func = nn.L1Loss()
        data = np.array(random.sample(self.buffer, 10))
        state = torch.from_numpy(data[:, 0:3]).float()
        action = torch.from_numpy(data[:, 3:4]).long()
        state_ = torch.from_numpy(data[:, 4:7]).float()
        reward = torch.from_numpy(data[:, 7:8]).float()

        q_eval = self.eval_net(state).gather(1, action)
        q_target = self.target_net(state_).detach()
        q_target = reward + gamma * q_target.max(1)[0].view(10, 1)
        loss = loss_func(q_eval, q_target)

        # print(q_eval)
        # print(q_target)
        # input()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.loss_list.append(loss.item())
        # print(loss.item())

    def valid(self):
        x = self.x
        k = self.k
        stride = self.stride
        f_target = self.f_target
        reward = 0
        for i in range(100):
            f = env.elasticity(x)
            e = f - f_target
            with torch.no_grad():
                action = self.eval_net(torch.from_numpy(np.array([x, f, e])).float())
                action = torch.argmax(action)
                if action.item() == 0:
                    k = k - stride
                    a = 0
                if action.item() == 1:
                    k = k + stride
                    a = 1
                if action.item() == 2:
                    k = k
                    a = 2
            x_ = env.pid(x, k, e)
            f_ = env.elasticity(x_)
            e_ = f_ - f_target
            r = -math.fabs(e_)
            if a == 2:
                r = r + 5
            reward = reward + r
            print('k=%f' % (k))
            print('%f, %f, %f' % (x, f, e))
            x = x_

    def soft_update(self):
        for target_param, eval_param in zip(self.target_net.parameters(), self.eval_net.parameters()):
            target_param.data.copy_(target_param * (1 - self.soft_update_rate) + eval_param * self.soft_update_rate)


class dqn_ed1():
    def __init__(self):
        self.x = 0
        self.k = 0
        self.stride = 0.01
        self.f_target = 10
        self.noise = 0

        self.eval_net = network.q_net()
        self.target_net = network.q_net()
        self.env_net = env_net()

        for target_param, eval_param in zip(self.target_net.parameters(), self.eval_net.parameters()):
            target_param.data.copy_(eval_param)

        self.lr = 0.001
        self.optim = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

        self.buffer = []
        self.buffer_len = 1000

        self.soft_update_rate = 0.01

        self.reward_list = []
        self.loss_list = []

    def save(self):
        torch.save(self.target_net.state_dict(), 'q_net.pth')

    def load(self):
        self.target_net.load_state_dict(torch.load('q_net.pth'))
        self.eval_net.load_state_dict(torch.load('q_net.pth'))

    def load_env_net(self):
        self.env_net.load_state_dict(torch.load('env_net.pth'))

    def pre_explore(self):
        x = self.x
        k = self.k
        stride = self.stride
        f_target = self.f_target
        for i in range(self.buffer_len):
            f = self.env_net(torch.unsqueeze(torch.tensor(x), dim=0).float().detach()).item() + (
                        0.5 - random.random()) * self.noise
            e = f - f_target
            if random.random() > 0.2:
                k = k
                a = 2
            else:
                if random.random() > 0.5:
                    k = k + stride
                    a = 1
                else:
                    k = k - stride
                    a = 0
            x_ = env.pid(x, k, e)
            f_ = self.env_net(torch.unsqueeze(torch.tensor(x_), dim=0).float().detach()).item() + (
                        0.5 - random.random()) * self.noise
            e_ = f_ - f_target
            r = -math.fabs(e_)
            if a == 2:
                r = r + 5
            self.buffer.append([x, f, e, a, x_, f_, e_, r])
            x = x_

    def explore(self, greedy_rate):
        x = self.x
        k = self.k
        stride = self.stride
        f_target = self.f_target
        reward = 0
        for i in range(100):
            f = self.env_net(
                torch.unsqueeze(torch.tensor(x), dim=0).float().detach()).item() + random.random() * self.noise
            e = f - f_target
            if random.random() > greedy_rate:
                with torch.no_grad():
                    action = self.eval_net(torch.from_numpy(np.array([x, f, e])).float())
                    action = torch.argmax(action)
                    a = action.item()
                    if a == 0:
                        k = k - stride
                    if a == 1:
                        k = k + stride
                    if a == 2:
                        k = k
            else:
                if random.random() > 0.2:
                    k = k
                    a = 2
                else:
                    if random.random() > 0.5:
                        k = k + stride
                        a = 1
                    else:
                        k = k - stride
                        a = 0
            x_ = env.pid(x, k, e)
            f_ = self.env_net(torch.unsqueeze(torch.tensor(x_), dim=0).float().detach()).item() + (
                        0.5 - random.random()) * self.noise
            e_ = f_ - f_target
            r = -math.fabs(e_)
            if a == 2:
                r = r + 5
            reward = reward + r
            if len(self.buffer) > self.buffer_len:
                self.buffer.pop(0)
            self.buffer.append([x, f, e, a, x_, f_, e_, reward])  # state, action, state_, reward
            x = x_
        self.reward_list.append(reward)

    def train(self):
        gamma = 0.9
        loss_func = nn.L1Loss()
        data = np.array(random.sample(self.buffer, 10))
        state = torch.from_numpy(data[:, 0:3]).float()
        action = torch.from_numpy(data[:, 3:4]).long()
        state_ = torch.from_numpy(data[:, 4:7]).float()
        reward = torch.from_numpy(data[:, 7:8]).float()

        q_eval = self.eval_net(state).gather(1, action)
        q_target = self.target_net(state_).detach()
        q_target = reward + gamma * q_target.max(1)[0].view(10, 1)
        loss = loss_func(q_eval, q_target)

        # print(q_eval)
        # print(q_target)
        # input()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.loss_list.append(loss.item())
        # print(loss.item())

    def valid(self):
        x = self.x
        k = self.k
        stride = self.stride
        f_target = self.f_target
        reward = 0
        for i in range(100):
            f = self.env_net(torch.unsqueeze(torch.tensor(x), dim=0).float().detach()).item()
            e = f - f_target
            with torch.no_grad():
                action = self.eval_net(torch.from_numpy(np.array([x, f, e])).float())
                action = torch.argmax(action)
                if action.item() == 0:
                    k = k - stride
                    a = 0
                if action.item() == 1:
                    k = k + stride
                    a = 1
                if action.item() == 2:
                    k = k
                    a = 2
            x_ = env.pid(x, k, e)
            f_ = self.env_net(torch.unsqueeze(torch.tensor(x_), dim=0).float().detach()).item()
            e_ = f_ - f_target
            r = -math.fabs(e_)
            if a == 2:
                r = r + 5
            reward = reward + r
            file_to_write = open('validation.txt', 'a')
            str_to_write = str(k) + ' ' + str(x) + ' ' + str(f) + ' ' + str(e) + '\n'
            file_to_write.write(str_to_write)
            file_to_write.close()
            x = x_
            print(a)

    def soft_update(self):
        for target_param, eval_param in zip(self.target_net.parameters(), self.eval_net.parameters()):
            target_param.data.copy_(target_param * (1 - self.soft_update_rate) + eval_param * self.soft_update_rate)


if __name__ == '__main__':
    demo = dqn_ed1()
    # demo.load()
    demo.load_env_net()

    demo.pre_explore()

    for i in range(1000):
        demo.explore(greedy_rate=(0.2 + 0.2 * (1000 - i) / 1000))
        for j in range(20):
            demo.train()
        demo.soft_update()
        print(i)
        print(demo.reward_list[len(demo.reward_list) - 1])
    #
    # for i in range(1000):
    #     demo.explore(greedy_rate=0.2)
    #     for j in range(20):
    #         demo.train()
    #     demo.soft_update()
    #     print(i)
    #     print(demo.reward_list[len(demo.reward_list) - 1])

    np.savetxt('reward.txt', np.array(demo.reward_list).transpose())
    np.savetxt('loss.txt', np.array(demo.loss_list).transpose())
    np.savetxt('buffer.txt', np.array(demo.buffer))
    demo.save()