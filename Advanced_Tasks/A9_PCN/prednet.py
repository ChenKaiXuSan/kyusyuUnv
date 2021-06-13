# %%
from operator import ne
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
# %%
# Feedforeward module 
class FFconv2d(nn.Module):
    def __init__(self, inchan, outchan, downsample=False):
        super().__init__()
        
        self.conv2d = nn.Conv2d(inchan, outchan, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = downsample
        if self.downsample:
            self.Downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv2d(x)
        if self.downsample:
            x = self.Downsample(x)
        return x 

# Feedback module 
class FBconv2d(nn.Module):
    def __init__(self, inchan, outchan, upsample=False):
        super().__init__()

        self.convtranspose2d = nn.ConvTranspose2d(inchan, outchan, kernel_size=3, stride=1, padding=1, bias=False)
        self.upsample = upsample
        if self.upsample:
            self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        if self.upsample:
            x = self.Upsample(x)
        x = self.convtranspose2d(x)
        return x 

# %%
# PredNet 
class PredNet(nn.Module):

    def __init__(self, num_classes=10, cls=3):
        super().__init__()

        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # num of circles
        self.nlays = len(ics) #number of layers

        # Feedforward layers 
        self.FFconv = nn.ModuleList([
            FFconv2d(ics[i], ocs[i], downsample=sps[i]) for i in range(self.nlays)
        ])
        # Feedback layers 
        if cls > 0:
            self.FBconv = nn.ModuleList([
                FBconv2d(ocs[i], ics[i], upsample=sps[i]) for i in range(self.nlays)
            ])

        # update rate 
        self.a0 = nn.ParameterList([
            nn.Parameter(torch.zeros(1, ics[i], 1, 1) + 0.5 ) for i in range(1, self.nlays)
        ])
        self.b0 = nn.ParameterList([
            nn.Parameter(torch.zeros(1, ocs[i], 1, 1) + 1.0 ) for i in range(self.nlays)
        ])

        # Linear layer 
        self.linear = nn.Linear(ocs[-1], num_classes)

    def forward(self, x):

        # Feedforward
        xr = [F.relu(self.FFconv[0](x))]
        for i in range(1, self.nlays):
            xr.append(F.relu(self.FFconv[i](xr[i-1])))

        # Dynamic process 
        for t in range(self.cls):

            # Feedback prediction 
            xp = []
            for i in range(self.nlays-1, 0, -1):
                xp = [self.FBconv[i](xr[i])] + xp
                a0 = F.relu(self.a0[i-1]).expand_as(xr[i-1])
                xr[i-1] = F.relu(xp[0]*a0 + xr[i-1]*(1-a0))

            # Feedforward prediction error 
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(self.FFconv[0](x-self.FBconv[0](xr[0]))*b0 + xr[0])
            for i in range(1, self.nlays):
                b0 = F.relu(self.b0[i]).expand_as(xr[i])
                xr[i] = F.relu(self.FFconv[i](xr[i-1]-xp[i-1])*b0 + xr[i])

        # classifier 
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

# %%
if __name__ == '__main__':
    net = PredNet()
    print(net)