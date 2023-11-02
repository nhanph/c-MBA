import torch.nn as nn
import torch.nn.functional as F

class FCNet(nn.Module):
    def __init__(self, n_feature, n_output, n_hidden=(1000,1000,1000,1000)):
        super(FCNet, self).__init__()
        self.layers = nn.ModuleList()
        current_dim = n_feature
        for hdim in n_hidden:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, n_output))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = self.layers[-1](x)
        return out

class FCNetWithSoftmax(nn.Module):
    def __init__(self, n_feature, n_output, n_hidden=(1000,1000,1000,1000), sm_temp=1.):
        super(FCNetWithSoftmax, self).__init__()
        self.layers = nn.ModuleList()
        current_dim = n_feature
        for hdim in n_hidden:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, n_output))

        self.sm_temp = sm_temp

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = F.softmax(self.layers[-1](x)/self.sm_temp, dim=0)
        return out