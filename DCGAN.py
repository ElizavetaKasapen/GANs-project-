#importing important Libraries
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transform
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torchvision.utils import make_grid
import matplotlib.pyplot as plt



# Defining Variables
img_size = 64
batch_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
dataroot = "../img_align_celeba"

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

if __name__ == '__main__':


    real_batch = next(iter(dataloader_train))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(make_grid(real_batch[0], padding=2, normalize=True), (1, 2, 0)))
    plt.show()