import torch
import torch.nn as nn

# [모델 1] 작성하신 기존 Baseline 모델
class ResidualDynamicNet(nn.Module):
    def __init__(self, input_dim=16, output_dim=6):
        super(ResidualDynamicNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, output_dim)
        ) 
    def forward(self, x):
        return self.net(x)

# [모델 2] Skip-Connection 추가형 ResNet 모델
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))

class ResNetDynamicNet(nn.Module):
    def __init__(self, input_dim=16, output_dim=6, hidden_dim=128):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.block1 = ResidualBlock(hidden_dim)
        self.block2 = ResidualBlock(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h = self.in_proj(x)
        h = self.block1(h)
        h = self.block2(h)
        return self.out_proj(h)