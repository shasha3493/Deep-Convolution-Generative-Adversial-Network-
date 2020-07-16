# Importing Libraries
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
import glob

# Set the manual seed for reproducibility
manualSeed = 999
random.seed = manualSeed
torch.manual_seed(manualSeed)

dataroot = './celeba/'
workers = 0
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 50
lr = 0.0002
beta1 = 0.5
ngpu = 1

device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')

# Creating dataloader
dataset = dset.ImageFolder(root = dataroot, transform = transforms.Compose([transforms.Resize(image_size),
                                                                            transforms.CenterCrop(image_size),
                                                                            transforms.ToTensor(),
                                                                            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = workers)


# Function for weight initialization
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02)
  if classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)

# Generator model
class Generator(nn.Module):
  def __init__(self, ngpu):
    super(Generator, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(nn.ConvTranspose2d(nz, ngf*16, 4, 1, 0, bias = False),
                              nn.BatchNorm2d(ngf*16),
                              nn.ReLU(True),
                              # (ngf*8,4,4)

                              nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias = False),
                              nn.BatchNorm2d(ngf*8),
                              nn.ReLU(True),
                              # (ngf*4,8,8)

                              nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias = False),
                              nn.BatchNorm2d(ngf*4),
                              nn.ReLU(True),
                              # (ngf*2, 16, 16)

                              nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias = False),
                              nn.BatchNorm2d(ngf*2),
                              nn.ReLU(True),
                              # (ngf, 32, 32)

                              nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1, bias = False),
                              nn.Tanh()
                              # (nc, 64, 64)
                              )
    
  def forward(self, input):
      return(self.main(input))

# Move the model to device
netG = Generator(ngpu).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
    
# Initialize model parameters
netG.apply(weights_init)

# Discriminator model
class Discriminator(nn.Module):
  def __init__(self, ngpu):
    super(Discriminator, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(nn.Conv2d(nc, ndf*2, 4, 2, 1, bias = False),
                              nn.LeakyReLU(0.2, inplace = True),
                              
                              nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias = False),
                              nn.BatchNorm2d(ndf*4),
                              nn.LeakyReLU(0.2, inplace=True),
                              
                              nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias = False),
                              nn.BatchNorm2d(ndf*8),
                              nn.LeakyReLU(0.2, inplace = True),
                              
                              nn.Conv2d(ndf*8, ndf*16, 4, 2, 1, bias = False),
                              nn.BatchNorm2d(ndf*16),
                              nn.LeakyReLU(0.2, inplace=True),
                              
                              nn.Conv2d(ndf*16, 1, 4, 1, 0, bias = False))
    
  def forward(self, input):
      return(self.main(input))

# Move the model to device
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Initialize model parameters
netD.apply(weights_init)

# Initialize the Binary Cross Entropy Loss
criterion = nn.BCEWithLogitsLoss()

# Create a batch of latent vectors that we will use  to visualize the progression of generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish the convention of real and fake label during traininig
real_label = 1
fake_label = 0

# Setup optimizers for both generator and discriminator
optimizerD = optim.Adam(netD.parameters(), lr = lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = lr, betas=(beta1, 0.999))

# Commented out IPython magic to ensure Python compatibility.
# Training Loop

# Lists to keep track of the progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print('Start Training Loop')

# For each epoch
for epoch in range(num_epochs):
  torch.cuda.empty_cache()
  # For each data in the dataoader
  for i, data in enumerate(dataloader, 0):

    ###########################################
    # Update Discriminator
    # Maximize log(D(x)) + log(1 - D(G(z)))
    ###########################################

    # Train with all real batch
    netD.zero_grad()
    real_cpu = data[0].to(device)
    b_size = real_cpu.shape[0]
    label = torch.full((b_size, ), real_label, device=device)

    # Forward pass real batches through D
    output = netD(real_cpu).view(-1)
    # Calculate Loss
    errD_real = criterion(output, label)
    # Calculate the gradients
    errD_real.backward()
    D_x = torch.sigmoid(output).mean().item()

    # train with all fake batch
    # Generate a batch of latent vectors
    noise = torch.randn(b_size, nz, 1, 1, device=device)
    # Generate a batch of fake image with G
    fake = netG(noise)  

    label = torch.full((b_size, ), fake_label, device=device)
    
    # Classify fake batch with network D
    output = netD(fake.detach()).view(-1)
    
    # Calculate D's loss on all fake images
    errD_fake = criterion(output, label)
    # Calculate gradient
    errD_fake.backward()
    D_G_z1 = torch.sigmoid(output).mean().item()

    # Add the gradients
    errD = errD_real + errD_fake
    # Update D
    optimizerD.step()

    ###########################################
    # Update G network: maximize log(D(G(z))
    ###########################################

    netG.zero_grad()
    label.fill_(real_label) # Fake images are real for generator cost
    # Since we just updated D, perform another forward pass for all fake batches through D
    output = netD(fake).view(-1)
    errG = criterion(output, label)
    # Calculate gradients for both G and D but update only G
    errG.backward()
    optimizerG.step()
    D_G_z2 = torch.sigmoid(output).mean().item()

    # Output training stats
    if i % 50 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (epoch, num_epochs, i, len(dataloader),
                  errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    # Save Losses for plotting later
    G_losses.append(errG.item())
    D_losses.append(errD.item())

    # Check how the generator is doing by saving G's output on fixed_noise
    if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    iters += 1

  # Save the mddels' parameters
  torch.save(netG, './weights/generator'+str(epoch) + '.pth')
  torch.save(netD, './weights/discriminator'+str(epoch) + '.pth')
