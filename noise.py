import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage


class BlockPixel(nn.Module):
    """This class remove random pixel in the image"""
    def __init__(self, param=0.9):
        """args:
                param: float -> pourcent of the pixel to delete
        """
        super(BlockPixel, self).__init__()
        self.param = param
        self.r = None

    def forward(self, size, device='cuda:0'):
        """ ccompute the filter
            args:
                size: torch.size -> the size of the filter to return
                device: torch.device -> the device for the tensor to return
            return:
                filtre: tensor -> the filter
        """
        filtre = torch.rand(size)
        filtre = torch.where(filtre < self.param, torch.tensor(0), torch.tensor(1))
        filtre = torch.tensor(filtre, device=device, dtype=torch.float32, requires_grad=False)

        return filtre, 0


class Patch_block(nn.Module):
    """This class keep a patch from the image"""
    def __init__(self, taille):
        """args:
                taille: float -> size of the square to keep
        """
        super(Patch_block, self).__init__()
        self.taille = taille

    def forward(self, size, device='cuda:0'):
        """ ccompute the filter
            args:
                size: torch.size -> the size of the filter to return
                device: torch.device -> the device for the tensor to return
            return:
                filtre: tensor -> the filter
        """
        w = np.random.randint(0, high=(64 - self.taille), size=size[0])
        h = np.random.randint(0, high=(64 - self.taille), size=size[0])
        filtre = torch.ones(size)
        for i in range(size[0]):
            filtre[i, :, w[i]:w[i] + self.taille, h[i]:h[i] + self.taille] = 0
        filtre = torch.tensor(filtre, device=device, dtype=torch.float32, requires_grad=False)

        return filtre, 0


class Band_block(nn.Module):
    """This class remove a patch from the image"""
    def __init__(self, taille):
        """args:
                taille: float -> pourcent of the pixel to delete
        """
        super(Band_block, self).__init__()
        self.taille = taille

    def forward(self, size, device='cuda:0'):
        """ compute the filter
            args:
                 size: torch.size -> the size of the filter to return
                 device: torch.device -> the device for the tensor to return
            return:
                 filtre: tensor -> the filter
        """
        w = np.random.randint(0, high=(64 - self.taille), size=size[0])
        filtre = torch.ones(size)
        for i in range(size[0]):
            filtre[i, :, w[i]:w[i] + self.taille] = 0

        filtre = torch.tensor(filtre, device=device, dtype=torch.float32, requires_grad=False)

        return filtre, 0

class Rand_Block(nn.Module):
    """This class remove a patch wich is a random shape from the image"""
    def __init__(self, number):
        """args:
                taille: float -> pourcent of the pixel to delete
        """
        super(Rand_Block, self).__init__()
        self.n = number

    def forward(self, size, device='cuda:0'):
        """ compute the filter
            args:
                 size: torch.size -> the size of the filter to return
                 device: torch.device -> the device for the tensor to return
            return:
                 filtre: tensor -> the filter
        """

        filtre = np.zeros(size[-2:])
        points = size[-1] * np.random.random((2, self.n ** 2))
        filtre[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
        im = ndimage.gaussian_filter(filtre, sigma=size[-1] / (4. * self.n))

        mask = torch.where(torch.FloatTensor(im) > im.mean(), torch.tensor(0), torch.tensor(1)).expand(size)
        mask = torch.tensor(mask, device=device, dtype=torch.float32, requires_grad=False)
        return mask, 0
