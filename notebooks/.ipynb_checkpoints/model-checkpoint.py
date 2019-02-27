# General imports
from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from keijzer_exogan import *


class Generator(nn.Module):
    def __init__(self, ngpu, nz=100, ngf=32, nc=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.nc = nc
        self.ngf = ngf
        
        """
        where (in_channels, out_channels, 
        kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        """
        self.main = nn.Sequential(
            
            #1
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            #nn.BatchNorm2d(ngf * 8),
            #nn.LayerNorm(ngf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),
            
            #4
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 4),
            #nn.LayerNorm(ngf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),
            
            #7
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 2),
            #nn.LayerNorm(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),
            
            #10
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf*1, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf*1),
            #nn.LayerNorm(ngf*1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),
            
            # Go from 1x64x64 to 1x32x32
            nn.Conv2d(ngf, ngf, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf*1),
            #nn.LayerNorm(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),
            
                        #10
            # state size. (ngf*2) x 16 x 16
            #nn.ConvTranspose2d( ngf * 2, ngf*1, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf*1),
            #nn.ReLU(True),
            
            #13
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf*1, nc, 4, 2, 1, bias=False),
            #nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

    
class Discriminator(nn.Module):
    def __init__(self, ngpu, nc=1, ndf=32):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 1),
            #nn.LayerNorm(ndf*1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),
            
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 1, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 1),
            #nn.LayerNorm(ndf*1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),
            
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 1, ndf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 2),
            #nn.LayerNorm(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),
            
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            #nn.LayerNorm(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),
            
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 8),
            #nn.LayerNorm(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),
            
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


