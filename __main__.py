import torch
import torch.optim as optim
import argparse
import torchvision.transforms as transforms
import sys; sys.path.append('./network')
from SAGAN import CNetG, CNetD
from noise import BlockPixel, BlockPatch
from utils import *
from dataset import CelebADatasetNoise
from train import train
from utils import sauvegarde_arg


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=25, help="nb iterations for training")
parser.add_argument('--ndf', type=int, default=32, help="Base size of feature maps in discriminator")
parser.add_argument('--ngf', type=int, default=32, help="Base size of feature maps in generator")
parser.add_argument('--lrD', type=float, default=0.0004, help="Learning rate for the discriminator")
parser.add_argument('--lrG', type=float, default=0.0001, help="Learning rate for the generator")
parser.add_argument('--batch_size', type=int, default=64, help="Number of image per batch")
parser.add_argument('--save_file', type=str, default='./log/base', help="root where save result")
parser.add_argument('--load_file', type=str, default='/home/victor/dataset/img_align_celeba', help="root where load dataset")
parser.add_argument('--param', type=float, default=None, help="params for noise")
opt = parser.parse_args()
print(opt)
sauvegarde_arg(opt.save_file, opt)

###############
# init variable
###############


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG = CNetG(opt.ngf).to(device)
netD = CNetD(opt.ndf).to(device)
optimizerG = optim.Adam(netG.parameters(), opt.lrG, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), opt.lrD, betas=(0.5, 0.999))
noise_module = BlockPixel(opt.param)

###########
# Load Data
###########
dataset = CelebADatasetNoise(opt.load_file,
                              noise_module,
                              transforms.Compose([transforms.Resize(64),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

############
# Train
############
train(netG, netD, noise_module, optimizerD, optimizerG, dataloader, device, opt.save_file, opt.epoch)
