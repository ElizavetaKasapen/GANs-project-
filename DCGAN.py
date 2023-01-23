#importing important Libraries
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transform
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import matplotlib.pyplot as plt



# Defining Variables
img_size = 64
batch_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
dataroot = "../img_align_celeba"

# Initializing LOSS FUNCTION: BINARY CROSS ENTROPY BCE
loss_function = nn.BCELoss()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training our dataset, using resize and CenterCrop to avoid if there is any rectangle image
dataset_train = ImageFolder(dataroot,
                            transform=transform.Compose([
                                transform.Resize(img_size),
                                transform.CenterCrop(img_size),
                                transform.ToTensor(),
                                transform.Normalize(*stats)
                            ]))

# Training DataLoader with shuffling to ensure that data will be shuffled every epoch
#  and num_workers because we will use multiple cores

dataloader_train = DataLoader(dataset_train,
                              batch_size,
                              shuffle=True,
                              num_workers=2)



# Generator's Implementation: As the paper which is mentioned by prof. Mellacci
# Paper's name: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
Generator = nn.Sequential(
    #input 100 * 1024 * 3
    nn.ConvTranspose2d(100, 1024, 3, 1, 0, bias=False),
    nn.BatchNorm2d(1024),
    nn.ReLU(True),
    #output  3 * 3

    #input
    nn.ConvTranspose2d(1024, 512, 3, 2, 0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    #output  7*7

    #input
    nn.ConvTranspose2d(512, 256, 3, 1, 0, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    #output  14 * 14

    # input
    nn.ConvTranspose2d(256, 128, 3, 1, 0, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # output  28*28

    nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),

    nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
    nn.Tanh()
)

# Generating some fake images
# Random latent tensors
noise = torch.randn(batch_size, 100, 1, 1)
fake_imgs = Generator(noise)

# Training Generator:
def generator_train(gen_optimizer):
    gen_optimizer.zero_grad()
    # We will try fooling Discriminator
    predicted_fake_imgs = discriminator(fake_imgs)
    target_fake_imgs = torch.zeros(fake_imgs.size(0), 1)
    gen_loss = loss_function(predicted_fake_imgs, target_fake_imgs)

    #
    gen_loss.backward()
    gen_optimizer.step()

    return gen_loss.item()

# Discriminator's Implementation: As the paper which is mentioned by prof. Mellacci

discriminator = nn.Sequential(
    # I/P:

    nn.Conv2d(1, 64, 4, 2, 1, bias=False),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(1, 128, 4, 2, 1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(1, 256, 4, 2, 1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(256, 512, 4, 2, 1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(512, 1024, 3, 2, 0, bias=False),
    nn.BatchNorm2d(1024),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(1024, 1, 3, 1, 0, bias=False),
    # O/P : 1*1*1
    nn.Sigmoid()
)

def discriminator_train(real_imgs, disc_optimizer):
    # Training Real Images
    predicted_real_imgs = discriminator(real_imgs)
    target_real_imgs = torch.ones(real_imgs.size(0), 1)
    real_imgs_loss = loss_function(predicted_real_imgs, target_real_imgs)

    # Training Fake Images

    predicted_fake_imgs = discriminator(real_imgs)
    target_fake_imgs = torch.zeros(fake_imgs.size(0), 1)
    fake_imgs_loss = loss_function(predicted_fake_imgs, target_fake_imgs)

    disc_loss = real_imgs_loss + fake_imgs_loss
    disc_loss.backward()
    disc_optimizer.step()

    return disc_loss.item()




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_batch = next(iter(dataloader_train))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(make_grid(real_batch[0], padding=2, normalize=True), (1, 2, 0)))
    plt.show()
    plt.title("Fake Images")
    plt.imshow(np.transpose(make_grid(fake_imgs[0], padding=2, normalize=True), (1, 2, 0)))
    plt.show()
