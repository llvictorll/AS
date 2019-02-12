import torch
import torch.optim as optim
import argparse
import torchvision.transforms as transforms
import sys; sys.path.append('./network')
from torch.utils.data.sampler import SubsetRandomSampler
from SAGAN import CNetG, CNetD
from utils import choose_noise, loading_model
from dataset import CelebADatasetNoise
from train import train, pix2pix

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=25, help="nb iterations for training")
parser.add_argument('--ndf', type=int, default=64, help="Base size of feature maps in discriminator")
parser.add_argument('--ngf', type=int, default=64, help="Base size of feature maps in generator")
parser.add_argument('--lrD', type=float, default=0.0004, help="Learning rate for the discriminator")
parser.add_argument('--lrG', type=float, default=0.0001, help="Learning rate for the generator")
parser.add_argument('--batch_size', type=int, default=8, help="Number of image per batch")
parser.add_argument('--save_file', type=str, default='./log/base', help="root where save result")
parser.add_argument('--load_file', type=str, default='/home/victor/dataset/img_align_celeba', help="root where load dataset")
parser.add_argument('--noise', type=str, default="pixelblock", help="pixelblock|patchblock|bandblock|randblock")
parser.add_argument('--load_network', type=str, default="", help="load pretrain network start with")
parser.add_argument('--param', type=float, default=0.9, help="params for noise")
parser.add_argument('--alpha', type=float, default=2.0, help="hyper_alpha")
opt = parser.parse_args()

###############
# init variable
###############
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG = CNetG(opt.ngf).to(device)
netD = CNetD(opt.ndf).to(device)
optimizerG = optim.Adam(netG.parameters(), opt.lrG, betas=(0, 0.9))
optimizerD = optim.Adam(netD.parameters(), opt.lrD, betas=(0, 0.9))
noise_module = choose_noise(opt.noise, opt.param)
if opt.load_network != "":
    netG, netD, optimizerG, optimizerD = loading_model(netG, netD, optimizerG, optimizerD, device, opt.load_network)

###########
# Load Data
###########
train_dataset = CelebADatasetNoise("/home/victor/dataset/celebA_Train", noise_module,
                                   transforms.Compose([transforms.Resize(64),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ]))

val_dataset = CelebADatasetNoise("/home/victor/dataset/celebA_Val", noise_module,
                                  transforms.Compose([transforms.Resize(64),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                     ]))

test_dataset = CelebADatasetNoise("/home/victor/dataset/celebA_Test", noise_module,
                                  transforms.Compose([transforms.Resize(64),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                     ]))


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, shuffle=True)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, shuffle=False)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, drop_last=True, shuffle=False)

############
# Train
############
train(netG, netD, noise_module, optimizerD, optimizerG, train_loader, val_loader, device, opt.batch_size, opt.save_file, opt.epoch, opt.alpha)
