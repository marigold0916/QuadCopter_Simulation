import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ResidualDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

class ResidualDynamicNet(nn.Module):
    def __init__(self, input_dim = 16, output_dim = 6):
        super(ResidualDynamicNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128,128),
            nn.SiLU(),
            nn.Linear(128,64),
            nn.SiLU(),
            nn.Linear(64,output_dim)
        )
    def forward(self, x):
        return self.net(x)