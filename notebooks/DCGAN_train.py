# Imports
import random
import numpy as np
import time as t

import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

import model
from keijzer_exogan import *

# initialize random seeds
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)


"""
Local variables
"""
workers = 0 # Number of workers for dataloader, 0 when to_vram is enabled
batch_size = 16 # 2**11
image_size = 32
nz = 100 # size of latent vector
num_epochs =7*10**3
torch.backends.cudnn.benchmark=True # Uses udnn auto-tuner to find the best algorithm to use for your hardware, speeds up training by almost 50%
lr = 2e-4
lr_G = 2e-4
beta1 = 0.5 # Beta1 hyperparam for Adam optimizers
selected_gpus = [0] # Number of GPUs available. Use 0 for CPU mode.

path = '/datb/16011015/ExoGAN_data/selection//' #notice how you dont put the last folder in here...
images = np.load(path+'first_chunks_25_percent_images.npy')

swap_labels_randomly = False

train_d_g_conditional = False # switch between training D and G based on set threshold
d_g_conditional_threshold = 0.49 # D_G_z1 < threshold, train G

train_d_g_conditional_per_epoch = False

use_saved_weights = False


print('Batch size: ', batch_size)
ngpu = len(selected_gpus)
print('Number of GPUs used: ', ngpu)


"""
Load data and prepare DataLoader
"""
shuffle = True

if shuffle:
    np.random.shuffle(images) # shuffles the images

images = images[:int(len(images)*0.1)] # use only first ... percent of the data (0.05)
print('Number of images: ', len(images))

dataset = numpy_dataset(data=images, to_vram=True) # to_vram pins it to all GPU's
#dataset = numpy_dataset(data=images, to_vram=True, transform=transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])) # to_vram pins it to all GPU's

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers, pin_memory=False)


"""
Load and setup models
"""
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
        netG.load_state_dict(torch.load('netG_state_dict2', map_location=device)) #net.module..load_... for parallel model , net.load_... for single gpu model
        netD.load_state_dict(torch.load('netD_state_dict2', map_location=device))
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
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999)) # should be sgd
optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, 0.999))

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []

switch = True # condition switch, to switch between D and G per epoch

train_D = True
train_G = True

"""
Train the model
"""
iters = 0

t1 = t.time()
print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    q = np.random.randint(3, 6)
    
    if train_d_g_conditional_per_epoch:
        if switch == True:
            train_D = True
            train_G = False
            switch = False
        else:
            train_G = True
            train_D = False
            switch = True
            
    for i, data in enumerate(dataloader, 0):
        real_cpu = data.to(device) # for PIL images
        b_size = real_cpu.size(0)

        """
        https://github.com/soumith/ganhacks
        implement random label range from 0.0-0.3 to 0.7-1.2 for fake and real respectively
        """
        low = 0.01
        high = 0.2 #0.3
        fake_label = (low - high) * torch.rand(1) + high # uniform random dist between low and high
        fake_label = fake_label.data[0] # Gets the variable out of the tensor

        low = 0.8 #0.7
        high = 1.0
        real_label = (low - high) * torch.rand(1) + high # uniform random dist between low and high
        real_label = real_label.data[0] # Gets the variable out of the tensor

        label = torch.full((b_size,), real_label, device=device)
        
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)

        if swap_labels_randomly:
            if i % q == 0:
                labels_inverted = 'yes' 
                label.fill_(fake_label)
            else:
                labels_inverted = 'no'
        
        label.fill_(real_label)
        labels_inverted = 'no'
        label.fill_(real_label)
        
        if train_d_g_conditional:
            if i > 1:
                if D_G_z1 < d_g_conditional_threshold: # 45
                    train_G = True
                    train_D = False
                else:
                    train_D = True
                    train_G = False
                    
        if train_D:
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            
        # Forward pass real batch through D
        #print('real_cpu shape: ', real_cpu.shape)
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label) ## make this fake label sometimes
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)

        #if i % 11: # show first image of the real dataset every ... iterations
        #    print(fake.shape)
        #    plt.imshow(fake.reshape(32, 32))
        #    plt.show()
        
        # swap labels for the discriminator when i % q == 0 (so once every q-th batch)
        #if i % q == 0:
        #    label.fill_(real_label)
        #else:
        #    label.fill_(fake_label)

        label.fill_(fake_label) ## make this real label sometimes
        
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        
        if train_D:
            optimizerD.step()
        
        if train_G:
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        
        if train_G:
            # Update G
            optimizerG.step()
                
        t2 = t.time()
        # Output training stats
        
        if train_G and train_D:
            training_dg = 'D & G'
        elif train_G:
            training_dg = 'G'
        elif train_D:
            training_dg = '\t D'
        
        if iters % (20) == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\t Time: %.2f \t q: %s \t training: %s'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, (t2-t1), q, training_dg))
            t1 = t.time()

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 20 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
        if (iters % 100 == 0): # save weights every % .... iters
            if ngpu > 1:
                torch.save(netG.module.state_dict(), 'netG_state_dict2')
                torch.save(netD.module.state_dict(), 'netD_state_dict2')
                print('weights saved')
            else:
                torch.save(netG.state_dict(), 'netG_state_dict2')
                torch.save(netD.state_dict(), 'netD_state_dict2')
                print('weights saved')

        iters += 1