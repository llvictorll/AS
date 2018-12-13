import scipy.misc
import torchvision.utils as vutils
import numpy as np
import os

base_file= "/home/victor/PycharmProject/AS/project/log/"

def imshow(img):
    img = img / 2 + 0.5
    npimg = (img.cpu()).numpy()
    return np.transpose(npimg, (1, 2, 0))


def torch2PIL(img):
    img = img / 2 + 0.5
    img = np.transpose(img, (1, 2, 0))
    img = np.array(img*255, dtype='uint8')
    return img


def printG(x, k, netG, file):
    if not os.path.exists(file):
        os.makedirs(file)
    o = netG(x)
    scipy.misc.imsave(file+'/g{}.png'.format(k), imshow(vutils.make_grid(o.data)))


def print_img(x, name, file):
    if not os.path.exists(file):
        os.makedirs(file)
    scipy.misc.imsave( file + '/' + name + '.png', imshow(vutils.make_grid(x).data))


def sauvegarde_init(file):
    if not os.path.exists(file):
        os.makedirs(file)
    with open(file+"/res.csv", 'a') as f:
        f.write('dTrue' + '\t' + 'dFalse' + '\t' + 'qualité_test' + '\t' + 'qualité_train' + '\t' + 'référence' + '\n')


def sauvegarde(file, *agr):
    with open(file+"/res.csv", 'a') as f:
        for a in agr:
            f.write(str(a) + '\t')
        f.write('\n')


def save_net(file, *args):
    with open(+ file + "/net.txt", 'a') as f:
        for a in args:
            f.write(str(a)+'\n')


def urandom():
    return np.random.randint(10, size=10)+1

