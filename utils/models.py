# this script contains several prebuilt architectures that can be used to predict eps_0
# note that any model that performs regression can be used as the architecture underneath a diffusion model

import torch.nn as nn
import torch.nn.functional as F
import torch


class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size


class Model3layerPosEnc_1D(nn.Module):  # for one-dimensional data
    def __init__(self):
        super(Model3layerPosEnc_1D, self).__init__()
        self.time_emb = SinusoidalEmbedding(128)
        self.space_emb = SinusoidalEmbedding(128, scale=25.0)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.sigma = nn.GELU()

    def forward(self, x_org):
        # embedding
        x, t = x_org[:, 0], x_org[:, 1]
        x_emb = self.space_emb(x)
        t_emb = self.time_emb(t)
        x = torch.cat((x_emb, t_emb), dim=-1)
        # model
        z1 = self.sigma(self.fc1(x))
        z2 = z1 + self.sigma(self.fc2(z1))
        s = self.fc3(z2)
        return s + x_org[:, 0].unsqueeze(1)


class Model3layerPosEnc_2D(nn.Module):  # for two-dimensional data
    def __init__(self):
        super(Model3layerPosEnc_2D, self).__init__()
        self.time_emb = SinusoidalEmbedding(128)
        self.space_emb = SinusoidalEmbedding(128, scale=25.0)
        self.fc1 = nn.Linear(384, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.sigma = nn.GELU()

    def forward(self, x_org):
        # embedding
        x1, x2, t = x_org[:, 0], x_org[:, 1], x_org[:, 2]
        x1_emb = self.space_emb(x1)
        x2_emb = self.space_emb(x2)
        t_emb = self.time_emb(t)
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        # model
        z1 = self.sigma(self.fc1(x))
        z2 = z1 + self.sigma(self.fc2(z1))
        s = self.fc3(z2)
        return s + x_org[:, :2]


class Model3layerPosEnc_CC(nn.Module):  # model for credit card dataset (29 dimensions)
    def __init__(self):
        super(Model3layerPosEnc_CC, self).__init__()
        self.time_emb = SinusoidalEmbedding(128)
        self.amount_emb = SinusoidalEmbedding(128, scale=25.0)
        self.fc1 = nn.Linear(284, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 29)
        self.sigma = nn.GELU()

    def forward(self, x_org):
        # embedding
        amount, t, v = x_org[:, -2], x_org[:, -1], x_org[:, :-2]
        x_emb = self.amount_emb(amount)
        t_emb = self.time_emb(t)
        x = torch.cat((v, x_emb, t_emb), dim=-1)
        # model
        z1 = self.sigma(self.fc1(x))
        z2 = z1 + self.sigma(self.fc2(z1))
        s = self.fc3(z2)
        return s + x_org[:, :-1]
