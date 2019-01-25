import torch
import torch.nn.functional as F
import numpy as np

from utils import save_net, sauvegarde_init, print_img, sauvegarde, printG
from tqdm import tqdm
from dataset import *
from collections import deque


def train(netG, netD, noise_module, optimizerD, optimizerG, trainloader, valloader, device, batch_size, file, epoch,
          alpha):
    print(file)
    save_net(file, netG, netD)
    sauvegarde_init(file, "train")
    sauvegarde_init(file, "eval")
    netG.train()
    netD.train()
    cpt = 0
    dTrue = deque(maxlen=1000)
    dFalse = deque(maxlen=1000)
    mse_train = deque(maxlen=1000)
    mse_val = deque(maxlen=1000)
    turn = True
    filtre_size = (batch_size, 3, 64, 64)

    bar_epoch = tqdm(range(epoch))
    bar_data = tqdm(range(len(trainloader)))
    for e in bar_epoch:
        for i, (ref, y) in zip(bar_data, trainloader):
            real_label = torch.FloatTensor(ref.size(0)).fill_(.9).to(device)
            fake_label = torch.FloatTensor(ref.size(0)).fill_(.1).to(device)

            ref = ref.to(device)
            y = y.to(device)
            if i % 3 in [0, 1]:
                ################
                # train D
                ################
                optimizerD.zero_grad()

                # avec de vrais labels
                outputTrue = netD(y)
                lossDT = F.binary_cross_entropy_with_logits(outputTrue, real_label)

                # avec de faux labels
                x_hat = netG(y).detach()
                filtreD = noise_module.forward(filtre_size, device)
                y_hat = x_hat * filtreD
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
                filtreG = noise_module.forward(filtre_size, device)
                x_hat = netG(y)
                y_hat = filtreG * x_hat
                outputDbruit = netD(y_hat)
                lossBruit = F.binary_cross_entropy_with_logits(outputDbruit, real_label)

                # imporve G with MSE
                x_tilde = netG(y_hat)
                y_tilde = filtreG * x_tilde.detach()

                lossSupervise = F.mse_loss(y_hat, y_tilde)

                (alpha * lossSupervise + lossBruit).backward()
                optimizerG.step()

            # test
            mse_train.append(F.mse_loss(x_hat, ref).data)
            bar_data.set_postfix({"qual": np.array(mse_train).mean()})

            sauvegarde(file, "train", np.array(dTrue).mean(), np.array(dFalse).mean(), np.array(mse_train).mean())

            if i % 250 == 1:

                cpt += 1
                netG.eval()
                bar_test = tqdm(range(len(valloader)))
                for j, (ref, y) in zip(bar_test, valloader):
                    if turn:
                        save_xb = y.to(device)
                        print_img(save_xb, 'image_de_base_bruit', file)
                        print_img(ref, 'image_de_base_sans_bruit', file)
                        turn = False
                    ref = ref
                    y = y.to(device)

                    #############
                    # eval G
                    #############
                    img_gen = netG(y).detach().cpu()

                    mse_val.append(F.mse_loss(ref, img_gen))
                    sauvegarde(file, "eval", np.array(mse_val).mean())

                printG(save_xb, cpt, netG, file)
                netG.train()

            if i % 400 == 0 and i > 0:
                for g in optimizerD.param_groups:
                    g['lr'] *= 0.995
                for g in optimizerG.param_groups:
                    g['lr'] *= 0.995
