import torch
import torch.nn.functional as F
import numpy as np

from utils import save_net, sauvegarde_init, print_img, sauvegarde, printG
from tqdm import tqdm
from dataset import *


def train(netG, netD, noise_module, optimizerD, optimizerG, dataloader, device, file, epoch, alpha):
    print(file)
    save_net(file, netG, netD)
    sauvegarde_init(file)
    netG.train()
    netD.train()
    cpt = 0
    dTrue, dFalse, mse = [0], [0], [0]
    turn = True
    alpha = 0
    bar_epoch = tqdm(range(epoch))
    bar_data = tqdm(range(len(dataloader)))
    for _ in bar_epoch:
        for i, (ref, y) in zip(bar_data, dataloader):
            if turn:
                save_xb = y
                print_img(save_xb, 'image_de_base_bruit', file)
                print_img(ref, 'image_de_base_sans_bruit', file)
                turn = False
                continue

            real_label = torch.FloatTensor(ref.size(0)).fill_(.9).to(device)
            fake_label = torch.FloatTensor(ref.size(0)).fill_(.1).to(device)

            ref = ref.to(device)
            y = y.to(device)

            if i % 2 == 0:
                ################
                # train D
                ################
                optimizerD.zero_grad()

                # avec de vrais labels
                outputTrue = netD(y)
                lossDT = F.binary_cross_entropy_with_logits(outputTrue, real_label)

                # avec de faux labels
                x_hat = netG(y).detach()
                y_hat = noise_module(x_hat)
                outputFalse = netD(y_hat)
                lossDF = F.binary_cross_entropy_with_logits(outputFalse, fake_label)
                (lossDF + lossDT).backward()
                optimizerD.step()

                dTrue.append(torch.sigmoid(outputTrue).data.mean())
                dFalse.append(torch.sigmoid(outputFalse).data.mean())
                bar_epoch.set_postfix({"D(x)": np.array(dTrue).mean(), "D(G(x))": np.array(dFalse).mean()})

            else:
                #############
                # train G
                #############
                optimizerG.zero_grad()

                # improve G with Discriminator
                x_hat = netG(y)
                y_hat = noise_module(x_hat)
                outputDbruit = netD(y_hat)
                lossBruit = F.binary_cross_entropy_with_logits(outputDbruit, real_label)

                # improve G with MSE
                x_tilde = netG(y_hat)
                y_tilde = noise_module(x_tilde, pretrain=True)

                lossSupervise = F.mse_loss(y_hat, y_tilde)

                (lossBruit + alpha * lossSupervise).backward()
                optimizerG.step()

            #print
            mse.append(F.mse_loss(x_hat, ref))
            bar_data.set_postfix({"qual": np.array(mse).mean(), "cpt:": cpt})

            sauvegarde(file, np.array(dTrue).mean(), np.array(dFalse).mean(), np.array(mse).mean())

            if i % 100 == 0 and i != 0:
                alpha += 0.01
                alpha = min(alpha, 2)

            if i % 250 == 1:
                printG(save_xb, cpt, netG, file)
                cpt += 1
                dTrue, dFalse, mse, ref = [0], [0], [0], [0]
