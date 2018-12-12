from PIL import Image
import os
from noise import *
import torch
import torchvision.transforms as transforms

class CelebADatasetNoise(torch.utils.data.Dataset):
    """Load CelebA Dataset"""
    def __init__(self, imgFolder, f_bruit,transform=transforms.ToTensor()):
        super(CelebADatasetNoise, self).__init__()
        self.imgFolder = imgFolder
        self.transform = transform
        self.f_bruit = f_bruit
        self.list = os.listdir(self.imgFolder)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        """return:
            img: the real image
            imgb: the image with noise"""
        file = os.path.join(self.imgFolder,self.list[i])
        image = Image.open(file).crop((15,15,175,175))
        img = self.transform(image)
        if img.size(0) == 1:
            img = img.expand(3, img.size(1), img.size(2))
        imgb = self.f_bruit(img)
        return img, imgb

