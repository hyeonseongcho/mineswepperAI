import torch
import numpy as np

class Agent:
    def __init__(self, sizeX, sizeY, l_rate, gpu):
        self.gpu = gpu

        self.sizeQ = 2 * sizeX * sizeY

        self.mainDQ = DQN(sizeX, sizeY).to(self.gpu)
        self.tarDQ  = DQN(sizeX, sizeY).to(self.gpu)
        self.tarDQ.load_state_dict(self.mainDQ.state_dict())
        self.tarDQ.eval()

        self.criterion = torch.nn.MSELoss()

        self.opt_mainDQ = torch.optim.Adam(self.mainDQ.parameters(), l_rate)
        
    def update(self):
        self.tarDQ.load_state_dict(self.mainDQ.state_dict())

    def choose(self, eps, state, mask, learn=True):

        if learn: # e-greedy

            Q = self.mainDQ(state)

            if eps > torch.rand([1]):
                pos = torch.nonzero(mask == 0).flatten()
                if (len(pos) > 0):
                    tmp = torch.randint(0, len(pos), [1])
                    num = pos[tmp]
                else:
                    num = torch.randint(0, self.sizeQ, [1]).to(self.gpu)
            else:
                num = torch.argmax(Q - 10000*mask).unsqueeze(dim=0).to(self.gpu)

        else: # greedy
            Q = self.mainDQ(state)
            num = torch.argmax(Q - 10000*mask)

        return num, Q


class DQN(torch.nn.Module):
    def __init__(self, sizeX, sizeY):
        super().__init__()

        self.sizeX = sizeX
        self.sizeY = sizeY

        # conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_1 = torch.nn.Conv2d(1, 32, 3, 1, 1)
        self.conv_2 = torch.nn.Conv2d(32, 64, 3, 1, 1)
        self.conv_3 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_4 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_5 = torch.nn.Conv2d(64, 2, 3, 1, 1)

        self.relu = torch.nn.ReLU(True)

    def forward(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x = self.relu(self.conv_4(x))
        x = self.relu(self.conv_5(x))
        x = torch.flatten(x, 1)
        #x = x.view(-1)
        return x