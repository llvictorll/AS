import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BlockPixel():
    def __init__(self, param=0.9):
        super(BlockPixel, self).__init__()
        self.param = param
        self.r = None

    def forward(self, size, device):
        self.r = torch.rand(size)
        self.r = torch.where(self.r < self.param, torch.tensor(0), torch.tensor(1))
        self.r = torch.tensor(self.r, device=device, dtype=torch.float32, requires_grad=False)

        return self.r


class Patch_block(nn.Module):
    def __init__(self, taille):
        super(Patch_block, self).__init__()
        self.taille = taille

    def forward(self, size, device='cuda:0'):
        w = np.random.randint(0, high=(64 - self.taille), size=size[0])
        h = np.random.randint(0, high=(64 - self.taille), size=size[0])
        self.r = torch.ones(size)
        for i in range(size[0]):
            self.r[i, :, w[i]:w[i] + self.taille, h[i]:h[i] + self.taille] = 0
        self.r = torch.tensor(self.r, device=device, dtype=torch.float32, requires_grad=False)

        return self.r


class Band_block(nn.Module):
    def __init__(self, taille):
        super(Band_block, self).__init__()
        self.taille = taille

    def forward(self, size, device='cuda:0'):
        w = np.random.randint(0, high=(64 - self.taille), size=size[0])
        self.r = torch.ones(size)
        for i in range(size[0]):
            self.r[i, :, w[i]:w[i] + self.taille] = 0  # self.r[:, w+w +self.taille]

        self.r = torch.tensor(self.r, device=device, dtype=torch.float32, requires_grad=False)

        return self.r


class ConvNoise(nn.Module):
    def __init__(self, conv_size, noise_variance):
        super().__init__()
        self.conv_size = conv_size
        self.noise_variance = noise_variance

    def forward(self, x, device='cpu'):
        x_measured = x.clone()
        noise = torch.randn_like(x_measured) * self.noise_variance
        noise = torch.tensor(noise, device=device, requires_grad=False)
        eps = torch.ones(1, 1, self.conv_size, self.conv_size, device=device) / (self.conv_size * self.conv_size)
        for i in range(3):
            x_measured[:, i:i + 1] = F.conv2d(x[:, i:i + 1], eps, stride=1, padding=self.conv_size // 2)
        x_measured = x_measured + noise

        return x_measured.clamp(-0.999, 0.9999)
