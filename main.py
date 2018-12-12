import torch
import torch.optim as optim
import torch.nn.functional as F
from network import *
from noise import F_bruit, Patch_block,
import torchvision.transforms as transforms
from utils import *

from sacred import Experiment

from tqdm import tqdm
from dataset import *

ex = Experiment('test')

@ex.config
def conf():
    device = 'cuda:0'
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    optimizerG = optim.Adam(netG.parameters(), 0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), 0.001, betas=(0.5, 0.999))
    f_bruit = F_bruit
    epoch = 100
    cuda = True
    param = None
    file = '/tmp'
    f = f_bruit(param)

    dataset = CelebADatasetNoise("/net/girlschool/besnier/CelebA_dataset/img_align_celeba",
                            f,
                            transforms.Compose([transforms.Resize(64),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=1, drop_last=True)


@ex.automain
def main(netG, netD, f_bruit, epoch, param, cuda,device, dataloader, optimizerG, optimizerD, file):
    print(file)
    save_net(file, netG, netD)
    netG.train()
    netD.train()
    sauvegarde_init(file)
    cpt = 0
    dTrue = []
    dFalse = []
    mse = []
    ref = []
    module_bruit = f_bruit(param).to(device)
    turn = True
    bar_epoch = tqdm(range(epoch))
    bar_data = tqdm(range(len(dataloader)))
    for _ in bar_epoch:
        turnn = True
        for i, (x, xb) in zip(bar_data, dataloader):
            if turn:
                save_xb = xb
                print_img(save_xb, 'image_de_base_bruit', file)
                print_img(x, 'image_de_base_sans_bruit', file)
                turn = False
                if cuda:
                    save_xb = save_xb.cuda()
            if turnn:
                turnn = False
                continue

            real_label = torch.FloatTensor(x.size(0)*9).fill_(.9).to(device)
            fake_label = torch.FloatTensor(x.size(0)*9).fill_(.1).to(device)

            xref = module_bruit(x, b=True)
            x2 = netG(xref)
            xb2 = module_bruit(x2)
            if cuda:
                xb = xb.cuda()
                xb2 = xb2.cuda()
                x = x.cuda()
                x2 = x2.cuda()

            # train D
            optimizerD.zero_grad()

            # avec de vrais labels
            outputTrue = netD(xb)
            lossDT = F.binary_cross_entropy_with_logits(outputTrue, real_label)

            # avec de faux labels
            fake = module_bruit(netG(xb), b=True)
            outputFalse = netD(fake)
            lossDF = F.binary_cross_entropy_with_logits(outputFalse, fake_label)
            (lossDF + lossDT).backward()
            optimizerD.step()

            # train G

            optimizerG.zero_grad()
            outputG = module_bruit(netG(xb), b=True)
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

            sauvegarde(file, np.array(dTrue).mean(), np.array(dFalse).mean(), np.array(mse).mean(), np.array(ref).mean())

            if i % 250 == 1:
                printG(save_xb, cpt, netG, file)
                cpt += 1
                dTrue = []
                dFalse = []
                mse = []
                ref = []