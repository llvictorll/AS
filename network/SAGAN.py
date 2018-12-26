import torch
from torch import nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

num_classes = 1


class Self_AttentionLayer(nn.Module):
    """ self_attention module to allow the Gan to capture long-range dependencies
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


class CNetG(nn.Module):
    def __init__(self, ngf, nc=3):
        super(CNetG, self).__init__()

        self.conv1 = spectral_norm(nn.Conv2d(nc, ngf * 1, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(ngf * 1, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv3 = spectral_norm(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv4 = spectral_norm(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False))

        self.convT1 = spectral_norm(nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False))
        self.convT2 = spectral_norm(nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False))
        self.convT3 = spectral_norm(nn.ConvTranspose2d(ngf * 2, ngf * 1, kernel_size=4, stride=2, padding=1, bias=False))
        self.convT4 = spectral_norm(nn.ConvTranspose2d(ngf * 1, nc, kernel_size=4, stride=2, padding=1, bias=False))

        self.att = Self_AttentionLayer(ngf * 1, ngf // 8)

        self.bn1 = nn.BatchNorm2d(ngf * 1)
        self.bn2 = nn.BatchNorm2d(ngf * 2)
        self.bn3 = nn.BatchNorm2d(ngf * 4)
        self.bn4 = nn.BatchNorm2d(ngf * 8)

        self.bn5 = nn.BatchNorm2d(ngf * 4)
        self.bn6 = nn.BatchNorm2d(ngf * 2)
        self.bn7 = nn.BatchNorm2d(ngf * 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(self.dropout(x))))
        x2 = F.relu(self.bn2(self.conv2(self.dropout(x1))))
        x3 = F.relu(self.bn3(self.conv3(self.dropout(x2))))
        x4 = F.relu(self.bn4(self.conv4(self.dropout(x3))))

        out = F.relu(self.bn5(self.convT1(x4)))
        out = F.relu(self.bn6(self.convT2(out+x3)))
        out = F.relu(self.bn7(self.convT3(out+x2)))
        out = self.att(out)
        out = torch.tanh(self.convT4(out))

        return out


class CNetD(nn.Module):
    def __init__(self, ndf, nc=3):
        super(CNetD, self).__init__()
        self.ndf = ndf

        self.conv1 = spectral_norm(nn.Conv2d(nc, ndf * 1, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(ndf * 1, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv3 = spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv4 = spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv5 = spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False))

        self.att = Self_AttentionLayer(ndf * 2, ndf // 4)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.att(x)
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.conv5(x)

        return x.squeeze()


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
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.att(x)
        x = F.relu(self.bn4(self.conv4(x)))
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

if __name__ == '__main__':
    x = torch.randn(64,3,64,64)
    netg = CNetG(32)
    netd = CNetD(32)
    a = netg(x)
    b = netd(x)
    print(a.size())
    print(b.size())
