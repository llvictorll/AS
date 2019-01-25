import scipy.misc
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import os
from noise import BlockPixel, Patch_block, Band_block

base_file = "/home/victor/PycharmProject/AS/project/base/"


def imshow(img):
    # img = img / 2 + 0.5
    npimg = (img.cpu()).numpy()
    return np.transpose(npimg, (1, 2, 0))


def torch2PIL(img):
    # img = img / 2 + 0.5
    img = np.transpose(img, (1, 2, 0))
    # img = np.array(img*256, dtype='uint8')
    return img


def printG(x, k, netG, file):
    if not os.path.exists(file):
        os.makedirs(file)
    o = netG(x)
    scipy.misc.imsave(file + '/g{}.png'.format(k), imshow(vutils.make_grid(o.data, padding=2, normalize=True)))


def print_img(x, name, file):
    if not os.path.exists(file):
        os.makedirs(file)
    scipy.misc.imsave(file + '/' + name + '.png', imshow(vutils.make_grid(x, padding=2, normalize=True).data))


def printf(img):
    img = vutils.make_grid(img, padding=2, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.axis("off")
    plt.show()


def sauvegarde_init(file, name):
    if not os.path.exists(file + name):
        os.makedirs(file + name)
    with open(file + name + "/res.csv", 'a') as f:
        f.write('dTrue' + '\t' + 'dFalse' + '\t' + 'qualité_test' + '\t' + 'qualité_train' + '\t' + 'référence' + '\n')


def sauvegarde(file, name, *agr):
    with open(file + name + "/res.csv", 'a') as f:
        for a in agr:
            f.write(str(a) + '\t')
        f.write('\n')


def save_net(file, *args):
    with open(file + "/net.txt", 'w') as f:
        for a in args:
            f.write(str(a) + '\n')


def urandom():
    return np.random.randint(10, size=10) + 1


def choose_noise(noise, param):
    if noise == "pixelblock":
        return BlockPixel(param)
    elif noise == "patchblock":
        return Patch_block(int(param))
    elif noise == "bandblock":
        return Band_block(int(param))
    # elif noise == "convnoise":
    #    return ConvNoise(args[0], args[1])