import torch
from torch import nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from utils import save_net, sauvegarde_init, print_img, sauvegarde, printG

num_classes = 1


class Self_AttentionLayer(nn.Module):
    """ self_attention module to allow the Gan to capture longue-range dependancies
        in images"""
    def __init__(self, input_size, output_size):
        super(Self_AttentionLayer, self).__init__()
        self.output_size = output_size
        self.f_layer = nn.Conv2d(input_size, output_size, 1, stride=1)
        self.g_layer = nn.Conv2d(input_size, output_size, 1, stride=1)
        self.h_layer = nn.Conv2d(input_size, input_size, 1, stride=1)

        self.sm_layer = nn.Softmax(dim=1)
        self.gamma = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        batch, channel, width, height = x.size()

        # compute attention map
        f = self.f_layer(x).view(batch, self.output_size, width * height)
        g = self.g_layer(x).view(batch, self.output_size, width * height)
        f = torch.transpose(f, 1, 2)
        sm = self.sm_layer(torch.bmm(f, g))

        # compute self attention feature maps
        h = self.h_layer(x).view(batch, channel, -1)
        out = self.gamma * torch.bmm(h, sm).view(batch, channel, width, height) + x

        return out


class NetG(nn.Module):
    def __init__(self, nz, ngf, nc=3):
        super(NetG, self).__init__()
        self.conv1 = spectral_norm(nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False))
        self.conv2 = spectral_norm(nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv3 = spectral_norm(nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv4 = spectral_norm(nn.ConvTranspose2d(ngf * 2, ngf * 1, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv5 = spectral_norm(nn.ConvTranspose2d(ngf * 1, nc, kernel_size=4, stride=2, padding=1, bias=False))

        self.att = Self_AttentionLayer(ngf * 2, ngf // 4)

        self.bn1 = nn.BatchNorm2d(ngf * 8)
        self.bn2 = nn.BatchNorm2d(ngf * 4)
        self.bn3 = nn.BatchNorm2d(ngf * 2)
        self.bn4 = nn.BatchNorm2d(ngf * 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.att(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = torch.tanh(self.conv5(x))

        return x


class NetD(nn.Module):
    def __init__(self, ndf, nc=3):
        super(NetD, self).__init__()
        self.ndf = ndf

        self.conv1 = spectral_norm(nn.Conv2d(nc, ndf * 1, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(ndf * 1, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv3 = spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv4 = spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv5 = spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False))

        self.att = Self_AttentionLayer(ndf * 4, ndf // 2)

        self.bn1 = nn.BatchNorm2d(ndf * 1)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.bn4 = nn.BatchNorm2d(ndf * 8)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.att(x)
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)

        return x.squeeze()


class CelebADataset(torch.utils.data.Dataset):
    """Load CelebA Dataset"""
    def __init__(self, imgFolder, transform=transforms.ToTensor()):
        super(CelebADataset, self).__init__()
        self.imgFolder = imgFolder
        self.transform = transform
        self.list = os.listdir(self.imgFolder)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        """return:
            img: the real image"""
        file = os.path.join(self.imgFolder,self.list[i])
        image = Image.open(file).crop((15,15,175,175))
        img = self.transform(image)
        if img.size(0) == 1:
            img = img.expand(3, img.size(1), img.size(2))
        return img

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    file = './log/gpu'
    netG = NetG(128, 32).to(device)
    netD = NetD(32).to(device)
    epoch = 100
    batch_size = 64
    optimizerG = optim.Adam(netG.parameters(), 0.0004, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), 0.0002, betas=(0.5, 0.999))
    noise_fixe = torch.randn(batch_size, 128, 1, 1).to(device)
    ###########
    # Load Data
    ###########
    dataset = CelebADataset('/home/victor/dataset/img_align_celeba',
                                 transforms.Compose([transforms.Resize(64),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                     ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

    print(file)
    save_net(file, netG, netD)
    sauvegarde_init(file)
    netG.train()
    netD.train()
    cpt = 0
    dTrue, dFalse = [], []
    turn = True

    bar_epoch = tqdm(range(epoch))
    bar_data = tqdm(range(len(dataloader)))

    for _ in bar_epoch:
        for i, x in zip(bar_data, dataloader):
            real_label = torch.FloatTensor(x.size(0)).fill_(.9).to(device)
            fake_label = torch.FloatTensor(x.size(0)).fill_(.1).to(device)
            z = torch.randn(x.size(0), 128, 1, 1).to(device)

            if i % 2 == 0:

                # train D
                optimizerD.zero_grad()

                # avec de vrais labels
                outputTrue = netD(x)
                lossDT = F.binary_cross_entropy_with_logits(outputTrue, real_label)

                # avec de faux labels
                fake = netG(z).detach()
                outputFalse = netD(fake)
                lossDF = F.binary_cross_entropy_with_logits(outputFalse, fake_label)
                (lossDF + lossDT).backward()
                optimizerD.step()

            else:
                # train G
                optimizerG.zero_grad()
                outputG = netG(z)
                outputD = netD(outputG)
                lossG = F.binary_cross_entropy_with_logits(outputD, real_label)
                lossG.backward()
                optimizerG.step()

            #test
            dTrue.append(torch.sigmoid(outputTrue).data.mean())
            dFalse.append(torch.sigmoid(outputFalse).data.mean())

            bar_epoch.set_postfix({"Dset": np.array(dTrue).mean(),
                                   "G": np.array(dFalse).mean()})

            sauvegarde(file, np.array(dTrue).mean(), np.array(dFalse).mean())

            if i % 100 == 1:
                printG(z, cpt, netG, file)
                cpt += 1
                dTrue, dFalse = [], []
