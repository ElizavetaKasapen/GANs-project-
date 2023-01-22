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

    nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
    nn.Tanh()
)


# Discriminator's Implementation: As the paper which is mentioned by prof. Mellacci

discriminator = nn.Sequential(
    # I/P: 3 * 64 * 64
    nn.Conv2d(1, 64, 4, 2, 1, bias=False),
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



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_batch = next(iter(dataloader_train))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(make_grid(real_batch[0], padding=2, normalize=True), (1, 2, 0)))
    plt.show()