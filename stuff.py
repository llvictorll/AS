import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('./network')

from utils import save_net, sauvegarde_init, print_img, sauvegarde, printG
from SAGAN import NetG, NetD
from noise import BlockPixel
from dataset import CelebADatasetNoise


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file = './log/gpu'
netG = NetG(128, 32).to(device)
netD = NetD(32).to(device)
epoch = 100
batch_size = 64
optimizerG = optim.Adam(netG.parameters(), 0.0004, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), 0.0002, betas=(0.5, 0.999))
noise_fixe = torch.randn(batch_size, 128, 1, 1).to(device)
module_bruit = BlockPixel(0.8)
###########
# Load Data
###########
dataset = CelebADatasetNoise('/home/victor/dataset/img_align_celeba',
                             module_bruit,
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
    for i, (x, xb) in zip(bar_data, dataloader):
        real_label = torch.FloatTensor(x.size(0)).fill_(.9).to(device)
        fake_label = torch.FloatTensor(x.size(0)).fill_(.1).to(device)

        if i % 2 == 0:

            # train D
            optimizerD.zero_grad()

            # avec de vrais labels
            outputTrue = netD(x)
            lossDT = F.binary_cross_entropy_with_logits(outputTrue, real_label)

            # avec de faux labels
            fake = netG(xb).detach()
            outputFalse = netD(fake)
            lossDF = F.binary_cross_entropy_with_logits(outputFalse, fake_label)
            (lossDF + lossDT).backward()
            optimizerD.step()

        else:
            # train G
            optimizerG.zero_grad()
            outputG = netG(xb)
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
            printG(xb, cpt, netG, file)
            cpt += 1
            dTrue, dFalse = [], []
