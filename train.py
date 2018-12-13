import torch
import torch.nn.functional as F
import numpy as np

from utils import save_net, sauvegarde_init, print_img, sauvegarde, printG
from tqdm import tqdm
from dataset import *



def train(netG, netD, noise_module, optimizerD, optimizerG, dataloader, device, opt):
    print(opt.file)
    save_net(opt.file, netG, netD)
    sauvegarde_init(opt.file)
    netG.train()
    netD.train()
    cpt = 0
    dTrue, dFalse, mse, ref = [], [], [], []
    turn = True

    bar_epoch = tqdm(range(opt.epoch))
    bar_data = tqdm(range(len(dataloader)))

    for _ in bar_epoch:
        for i, (x, xb) in zip(bar_data, dataloader):
            if turn:
                save_xb = xb
                print_img(save_xb, 'image_de_base_bruit', opt.file)
                print_img(x, 'image_de_base_sans_bruit', opt.file)
                turn = False
                continue

            real_label = torch.FloatTensor(x.size(0)*9).fill_(.9).to(device)
            fake_label = torch.FloatTensor(x.size(0)*9).fill_(.1).to(device)

            xref = noise_module(x).to(device)
            x2 = netG(xref).to(device)
            xb2 = noise_module(x2).to(device)

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
            lossSupervise = F.mse_loss(netG(xb2), x2)

            (lossBruit+lossSupervise).backward()
            optimizerG.step()

            #test
            dTrue.append(F.sigmoid(outputTrue).data.mean())
            dFalse.append(F.sigmoid(outputFalse).data.mean())
            mse.append(F.mse_loss(netG(xb).detach(), x))
            ref.append(F.mse_loss(x, F.upsample(xb, scale_factor=2)))

            bar_epoch.set_postfix({"Dset": np.array(dTrue).mean(),
                                   "G": np.array(dFalse).mean()})
            bar_data.set_postfix({"qual": np.array(mse).mean(),
                                  "ref": np.array(ref).mean()})

            sauvegarde(opt.file, np.array(dTrue).mean(), np.array(dFalse).mean(), np.array(mse).mean(), np.array(ref).mean())

            if i % 250 == 1:
                printG(save_xb, cpt, netG, opt.file)
                cpt += 1
                dTrue, dFalse, mse, ref = [], [], [], []