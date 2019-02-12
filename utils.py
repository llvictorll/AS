import scipy.misc
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from noise import BlockPixel, Patch_block, Band_block, Rand_Block

base_file = "/home/victor/PycharmProject/AS/project/base/"


def imshow(img):
    """transform the image before the plt.show..."""
    npimg = (img.cpu()).numpy()
    return np.transpose(npimg, (1, 2, 0))


def printG(x, k, netG, file):
    """save the image transformed by netG"""
    if not os.path.exists(file):
        os.makedirs(file)
    o = netG(x)
    scipy.misc.imsave(file + '/g{}.png'.format(k), imshow(vutils.make_grid(o.data, padding=2, normalize=True)))


def print_img(x, name, file):
    """save the x in the file"""
    if not os.path.exists(file):
        os.makedirs(file)
    scipy.misc.imsave(file + '/' + name + '.png', imshow(vutils.make_grid(x, padding=2, normalize=True).data))


def printf(img):
    """plot the image img"""
    img = vutils.make_grid(img, padding=2, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.axis("off")
    plt.show()

def sauvegarde_init(file, name):
    """init the file where we save the loss etc."""
    if not os.path.exists(file + name):
        os.makedirs(file + name)
    with open(file + name + "/res.csv", 'a') as f:
        f.write('dTrue' + '\t' + 'dFalse' + '\t' + 'qualité_test' + '\t' + 'qualité_train' + '\t' + 'référence' + '\n')


def sauvegarde(file, name, *agr):
    """ save the agr in the file"""
    with open(file + name + "/res.csv", 'a') as f:
        for a in agr:
            f.write(str(a) + '\t')
        f.write('\n')

def save_net(file, *args):
    """save the net in the file"""
    with open(file + "/net.txt", 'w') as f:
        for a in args:
            f.write(str(a) + '\n')

def save_model(netG, netD, optimizerG, optimizerD, epoch, file):
    """save the model G and D """
    torch.save({
        "generator":
            {
                'epoch': epoch + 1,
                'state_dict': netG.state_dict(),
                'optimizer': optimizerG.state_dict(),
            },
        "discriminator":
            {
                'epoch': epoch + 1,
                'state_dict': netD.state_dict(),
                'optimizer': optimizerD.state_dict(),
            }
    }, file + "/nets.pytorch")


def loading_model(netG, netD, optimizerG, optimizerD, device, file):
    """load the G and D from the file"""
    checkpoint = torch.load(file)
    netG.load_state_dict(checkpoint['generator']['state_dict'])
    netG.to(device)
    optimizerG.load_state_dict(checkpoint['generator']['optimizer'])
    netD.load_state_dict(checkpoint['discriminator']['state_dict'])
    netD.to(device)
    optimizerD.load_state_dict(checkpoint['discriminator']['optimizer'])
    return netG, netD, optimizerG, optimizerD


def choose_noise(noise, param):
    """choose the noise to use"""
    if noise == "pixelblock":
        return BlockPixel(param)
    elif noise == "patchblock":
        return Patch_block(int(param))
    elif noise == "bandblock":
        return Band_block(int(param))
    elif noise == "randblock":
        return Rand_Block(int(param))