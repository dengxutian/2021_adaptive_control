# @Time : 2021/8/3 10:04
# @Author : Deng Xutian
# @Email : dengxutian@126.com

import torch.nn as nn

class q_net(nn.Module):

    def __init__(self):
        super(q_net, self).__init__()
        self.net = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(3, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.net(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__ == '__main__':
    net = q_net()
    print(net)