import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class F_bruit(nn.Module):
    def __init__(self, param):
        super(F_bruit, self).__init__()
        self.param = param
        self.r = None

    def forward(self, x):
        self.r = torch.rand(x.size())
        self.r = np.where(self.r < self.param, 0, 1)
        if isinstance(x, torch.cuda.FloatTensor):
            self.r = torch.tensor(self.r, device='cuda', dtype=torch.float32, requires_grad=False)
        else:
            self.r = torch.tensor(self.r, device='cpu', dtype=torch.float32, requires_grad=False)

        return self.r * x


class Patch_block(nn.Module):
    def __init__(self, taille):
        super(Patch_block, self).__init__()
        self.taille = taille

    def forward(self, x):
        w = np.random.randint(0, 64 - self.taille)
        h = np.random.randint(0, 64 - self.taille)
        self.r = np.zeros(x.size())
        self.r[:, w:w + self.taille, h:h + self.taille] = 1
        if isinstance(x, torch.cuda.FloatTensor):
            self.r = torch.tensor(self.r, device='cuda', dtype=torch.float32, requires_grad=False)
        else:
            self.r = torch.tensor(self.r, device='cpu', dtype=torch.float32, requires_grad=False)
        return self.r * x


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
            x_measured[:, i:i + 1] = F.conv2d(x[:, i:i + 1], eps, stride=1, padding=self.conv_size//2)
        x_measured = x_measured + noise

        return x_measured.clamp(-0.999, 0.9999)
