"""
Discriminator and Generator implementation from DCGAN paper,
with removed Sigmoid() as output from Discriminator (and therefor
it should be called critic)
"""

import torch
import torch.nn as nn

class DcganD(nn.Module):
    """ dcgan Decriminator """
    def __init__(self, isize, nz, nc, ndf, n_extra_layers=0):
        super(DcganD, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        main.append(nn.Conv2d(nc, ndf, 4, 2, padding=1, bias=False))
        main.append(nn.LeakyReLU(0.2))

        csize, cndf = isize / 2, ndf

        # Extra layers
        for _ in range(n_extra_layers):
            main.append(nn.Conv2d(cndf, cndf, 3, 1, padding=1, bias=False))
            main.append(nn.BatchNorm2d(cndf))
            main.append(nn.LeakyReLU(0.2))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2

            main.append(nn.Conv2d(in_feat, out_feat, 4, 2, padding=1, bias=False))
            main.append(nn.BatchNorm2d(out_feat))
            main.append(nn.LeakyReLU(0.2))

            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.append(nn.Conv2d(cndf, 1, 4, 1, padding=0, bias=False))
        self.main = main

    def forward(self, input1):
        """construct"""
        output = self.main(input1)
        # output = output.mean(0)
        # return output.view(1)
        return output

class DcganG(nn.Module):
    """ dcgan generator """
    def __init__(self, isize, nz, nc, ngf, n_extra_layers=0):
        super(DcganG, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.append(nn.ConvTranspose2d(nz, cngf, 4, 1, padding=0, bias=False))
        main.append(nn.BatchNorm2d(cngf))
        main.append(nn.ReLU())

        csize = 4
        while csize < isize // 2:
            main.append(nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, padding=1, bias=False))
            main.append(nn.BatchNorm2d(cngf // 2))
            main.append(nn.ReLU())

            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for _ in range(n_extra_layers):
            main.append(nn.Conv2d(cngf, cngf, 3, 1, padding=1, bias=False))
            main.append(nn.BatchNorm2d(cngf))
            main.append(nn.ReLU())

        main.append(nn.ConvTranspose2d(cngf, nc, 4, 2, padding=1, bias=False))
        main.append(nn.Tanh())
        self.main = main

    def forward(self, input1):
        """construct"""
        output = self.main(input1)
        return output

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def main():
    batchSize, in_channels, H, W = 64, 3, 64, 64
    imageSize = 64
    nc = 3
    nz = 100
    ngf = 64
    ndf = 64
    n_extra_layers = 0
    x = torch.randn((batchSize, in_channels, H, W))
    disc = DcganD(imageSize, nz, nc, ndf, n_extra_layers)
    # assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    print(disc(x).shape)
    gen = DcganG(imageSize, nz, nc, ngf, n_extra_layers)
    z = torch.randn((batchSize, nz, 1, 1))
    # assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print(gen(z).shape)

if __name__ == '__main__':
    main()
