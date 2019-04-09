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
            # input is the latent vector z 100x1x1
            nn.ConvTranspose2d( nz, ngf * 32, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True), # Should use ReLU in generator according to DCGAN paper,
            
            #2
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            #3
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # G(z)
            nn.ConvTranspose2d( ngf * 8, nc, 4, 2, 1, bias=False),
            nn.Tanh() # Not used because ASPAs 
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
            #1
            # input is 1 x 32 x 32
            nn.Conv2d(nc, ndf*8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            #2
            nn.Conv2d(ndf*8, ndf * 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            #3
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(0.5),
            
            #4
            nn.Conv2d(ndf * 32, 1, 4, 3, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


