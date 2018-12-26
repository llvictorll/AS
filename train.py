import torch
import torch.nn.functional as F
import numpy as np

from utils import save_net, sauvegarde_init, print_img, sauvegarde, printG
from tqdm import tqdm
from dataset import *


def train(netG, netD, noise_module, optimizerD, optimizerG, dataloader, device, file, epoch):
    print(file)
    save_net(file, netG, netD)
    sauvegarde_init(file)
    netG.train()
    netD.train()
    cpt = 0
    dTrue, dFalse, mse, ref = [], [], [], []
    turn = True

    bar_epoch = tqdm(range(epoch))
    bar_data = tqdm(range(len(dataloader)))
    for _ in bar_epoch:
        for i, (x, xb) in zip(bar_data, dataloader):
            if turn:
                save_xb = xb
                print_img(save_xb, 'image_de_base_bruit', file)
                print_img(x, 'image_de_base_sans_bruit', file)
                turn = False
                continue


            real_label = torch.FloatTensor(x.size(0)).fill_(.9).to(device)
            fake_label = torch.FloatTensor(x.size(0)).fill_(.1).to(device)

            xref = noise_module(x).to(device)
            x2 = netG(xref).to(device)
            xb2 = noise_module(x2).to(device)
            x = x.to(device)
            xb = xb.to(device)

            # train D
            optimizerD.zero_grad()

            # avec de vrais labels
            outputTrue = netD(xb)
            lossDT = F.binary_cross_entropy_with_logits(outputTrue, real_label)

            # avec de faux labels
            fake = noise_module(netG(xb))
            outputFalse = netD(fake)
            lossDF = F.binary_cross_entropy_with_logits(outputFalse, fake_label)
            (lossDF + lossDT).backward()
            optimizerD.step()

            # train G
            optimizerG.zero_grad()
            outputG = noise_module(netG(xb))
            outputDbruit = netD(outputG)
            lossBruit = F.binary_cross_entropy_with_logits(outputDbruit, real_label)

            outputMSE = noise_module(netG(xb2))
            lossSupervise = F.mse_loss(outputMSE, noise_module(x2))

            (lossBruit+lossSupervise).backward()
            optimizerG.step()

            #test
            dTrue.append(torch.sigmoid(outputTrue).data.mean())
            dFalse.append(torch.sigmoid(outputFalse).data.mean())
            mse.append(F.mse_loss(netG(xb).detach(), x))

            bar_epoch.set_postfix({"D(x)": np.array(dTrue).mean(), "D(G(x))": np.array(dFalse).mean()})
            bar_data.set_postfix({"qual": np.array(mse).mean()})

            sauvegarde(file, np.array(dTrue).mean(), np.array(dFalse).mean(), np.array(mse).mean())

            if i % 250 == 1:
                printG(save_xb, cpt, netG, file)
                cpt += 1
                dTrue, dFalse, mse, ref = [], [], [], []
