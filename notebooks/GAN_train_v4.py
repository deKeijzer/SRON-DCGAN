# Imports
import random
import numpy as np
import time as t

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import gradcheck
import torchvision.utils as vutils
import time as time

from torch import autograd

import model_v4_small as model
import keijzer_exogan as ke

# initialize random seeds
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed) 
#torch.set_num_threads(16)

"""
Local variables
"""
selected_gpus = [0,1] # Selected GPUs

path = '/datb/16011015/ExoGAN_data/selection//' # Storage location of the train/test data

print('Loading data...')
images = np.load(path+'first_chunks_25_percent_images_v4.1.npy').astype('float32')

#images = images[:1000000] # select first ... images

use_saved_weights = False

g_iters = 1 # 5
d_iters = 2 # 1, discriminator is called critic in WGAN paper



"""
Local variables that generally stay unchanged
"""
batch_size = 64 # 64
num_epochs = 10*10**3

lrG = 1e-4
lrD = 1e-4

beta1 = 0.5 # beta1 for Adam
beta2 = 0.9 # beta2 for Adam

lambda_ = 10 # Scale factor for gradient_penalty

workers = 0 # Number of workers for dataloader, 0 when to_vram is enabled
image_size = 32
nz = 100 # size of latent vector
torch.backends.cudnn.benchmark=True # Uses udnn auto-tuner to find the best algorithm to use for your hardware, speeds up training by almost 50%


print('Batch size: ', batch_size)
ngpu = len(selected_gpus)
print('Number of GPUs used: ', ngpu)

"""
Load data and prepare DataLoader
"""
shuffle = True

if shuffle:
    np.random.shuffle(images) # shuffles the images

print('Number of images: ', len(images))

dataset = ke.numpy_dataset(data=images, to_vram=True) # to_vram pins it to all GPU's

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers, pin_memory=False)


"""
Load and setup models
"""
print('Initializing cuda...')
# Initialize cuda
device = torch.device("cuda:"+str(selected_gpus[0]) if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Load models
netG = model.Generator(ngpu).to(device)
netD = model.Discriminator(ngpu).to(device)

# Apply weights

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG.apply(weights_init) # It's not clean/efficient to load these ones first, but it works.
netD.apply(weights_init)

if use_saved_weights:
    try:
        # Load saved weights
        netG.load_state_dict(torch.load('gan_data//weights//netG_state_dict_gan_model_v4_small', map_location=device)) #net.module..load_... for parallel model , net.load_... for single gpu model
        netD.load_state_dict(torch.load('gan_data//weights//netD_state_dict_gan_model_v4_small', map_location=device))
        print('Succesfully loaded saved weights.')
    except:
        print('Could not load saved weights, using new ones.')
        pass

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, device_ids=selected_gpus, output_device=device)
    netD = nn.DataParallel(netD, device_ids=selected_gpus, output_device=device)
    
    
"""
Define input training stuff (fancy this up)
"""
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, beta2)) # should be sgd
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, beta2))

def save_progress(experiment_name, variable_name, progress_list):
    path = 'gan_data//training_progress//'
    progress_list = np.array(progress_list).astype('float32')
    #file = open(path+variable_name+"_"+experiment_name+'.npy', mode="a")
    file_name = path+variable_name+"_"+experiment_name+'.npy'
    file = np.load(file_name, mmap_mode='r+')
    file = np.append(file, progress_list)
    np.save(file_name, file)
    #file.close()


"""
Highly adapted from: https://github.com/jalola/improved-wgan-pytorch/blob/master/gan_train.py
"""
one = torch.FloatTensor([1]).to(device)
mone = one * -1

MSELoss = nn.MSELoss()
criterion = nn.BCELoss()

real_label = 1
fake_label = 0

iters = 0
t1 = time.time()
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        
        real = data.to(device)
        b_size = real.size(0)
        noise = torch.randn(b_size, nz, 1, 1, device=device)

        """
        Train D
        """
        for p in netD.parameters():
            p.requires_grad_(True)

        for _ in range(d_iters):
            label = torch.full((b_size,), real_label, device=device)
            
            netD.zero_grad()
            
            output = netD(real).view(-1)
            errD_real = criterion(output, label)
            
            errD_real.backward()
            D_x = output.mean().item()
            
            # train with fake batch
            
            fake = netG(noise)
            label.fill_(fake_label)
            
            output = netD(fake.detach()).view(-1)
            
            errD_fake = criterion(output, label)
            errD_fake.backward()
            
            d_real = output.mean().item()
            
            d_cost = errD_real + errD_fake
            
            optimizerD.step()
        
        """
        Train G
        """
        for p in netD.parameters():
            p.requires_grad_(False)
        
        # Calculate batch mean & std values, instead of using the mean/std of the complete train set.
        #real_mean = real.mean()
        #real_std = real.std()

        for _ in range(g_iters):
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            
            #noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            
            output = netD(fake).view(-1)
            
            #mean_L = MSELoss(output.mean(), real_mean)*1 # 3
            #std_L = MSELoss(output.std(), real_std)*1 # 3
            mean_L = 0
            std_L = 0
        
            g_cost = criterion(output, label)# - mean_L - std_L
            g_cost.backward()
            
            d_fake = output.mean().item()
            optimizerG.step()
        
        """
        
        """
        L = d_cost
        gradient_penalty = 0
        param_losses = 0
        
        
        """
        Save weights and print statistics
        """
        weights_saved = False
        if (iters % 100 == 0): # save weights every % .... iters
            #print('weights saved')
            if ngpu > 1:
                torch.save(netG.module.state_dict(), 'gan_data//weights//netG_state_dict_gan_model_v4_small')
                torch.save(netD.module.state_dict(), 'gan_data//weights//netD_state_dict_gan_model_v4_small')
            else:
                torch.save(netG.state_dict(), 'gan_data//weights//netG_state_dict_gan_model_v4_small')
                torch.save(netD.state_dict(), 'gan_data//weights//netD_state_dict_gan_model_v4_small')
            
        
        if i % (16) == 0:
            t2 = time.time()
            print('[%d/%d][%d/%d] \t Total loss = %.3f \t d_cost = %.3f \t g_cost = %.3f \t Gradient pen. = %.3f \t D(G(z)) = %.3f \t D(x) = %.3f \t mu L: %.3f \t std L: %.3f \t t = %.3f \t param_losses: %.3f'% 
                      (epoch, num_epochs, i, len(dataloader), L, d_cost, g_cost, gradient_penalty, d_fake, d_real, mean_L, std_L, (t2-t1), param_losses ))
            t1 = time.time()
            
        """Progress saver"""
        if iters == 0:
            arr_d_fake = []
            arr_d_real = []
            
        if (iters % (64) == 0) and (iters != 0):
            variable_names = ['d_fake', 'd_real']
            variables_to_save = [arr_d_fake, arr_d_real]
            
            for z,variable in enumerate(variables_to_save):
                save_progress('v4_test_gan', variable_names[z], variable)
            
            arr_d_fake = []
            arr_d_real = []
            print('saved variable progress')
        
        arr_d_fake.append(d_fake)
        arr_d_real.append(d_real)
        
        #print(len(arr_d_fake), len(arr_d_real))
        
        iters += i