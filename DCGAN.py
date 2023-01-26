#importing important Libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from tqdm.notebook import tqdm


# Setting Device:
'''if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
my_device = torch.device('cuda:0')'''
# Defining Variables
img_size = 64
batch_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
dataroot = "../img_align_celeba"

# Initializing LOSS FUNCTION: BINARY CROSS ENTROPY BCE
loss_function = nn.BCELoss()


'''device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''

# Training our dataset, using resize and CenterCrop to avoid if there is any rectangle image
dataset_train = ImageFolder(dataroot,
                            transform=transform.Compose([
                                transform.Resize(img_size),
                                transform.CenterCrop(img_size),
                                transform.ToTensor(),
                                transform.Normalize(*stats)
                            ]))



# real_batch = next(iter(dataloader_train))


# Generator's Implementation: As the paper which is mentioned by prof. Mellacci
# Paper's name: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
generator = nn.Sequential(
    #input 100 * 1024 * 3
    nn.ConvTranspose2d(100, 1024, 4, 1, 0, bias=False),
    nn.BatchNorm2d(1024),
    nn.ReLU(True),
    #output  3 * 3

    #input
    nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    #output  7*7

    #input
    nn.ConvTranspose2d(512, 256, 4, 1, 0, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    #output  14 * 14

    # input
    nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # output  28*28

    nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),

    nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
    nn.Tanh()
)


# Training Generator:
def generator_train(gen_optimizer):
    gen_optimizer.zero_grad()
    # Generating some fake images
    # Random latent tensors
    noise = torch.randn(batch_size, 100, 1, 1)
    fake_imgs = generator(noise)

    # We will try fooling Discriminator
    predicted_fake_imgs = discriminator(fake_imgs)
    target_fake_imgs = torch.ones(batch_size, 1)
    loss = loss_function(predicted_fake_imgs, target_fake_imgs)

    #
    loss.backward(retain_graph=True)
    gen_optimizer.step()

    return loss.item()

# Discriminator's Implementation: As the paper which is mentioned by prof. Mellacci

discriminator = nn.Sequential(
    # I/P:
    nn.Conv2d(3, 64, 4, 2, 1, bias=False),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(64, 128, 4, 2, 1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(128, 256, 4, 2, 1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(256, 512, 4, 2, 1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
    nn.BatchNorm2d(1024),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(1024, 1, 4, 2, 1, bias=False),
    # O/P : 1*1*1
    nn.Flatten(),
    nn.Sigmoid()
)

def discriminator_train(real_imgs, disc_optimizer):
    disc_optimizer.zero_grad()
    # Training Real Images
    predicted_real_imgs = discriminator(real_imgs)
    target_real_imgs = torch.ones(real_imgs.size(0), 1)
    real_imgs_loss = loss_function(predicted_real_imgs, target_real_imgs)
    real_avg = torch.mean(predicted_real_imgs).item()
    # Random latent tensors
    noise = torch.randn(batch_size, 100, 1, 1)
    fake_imgs = generator(noise)

    # Training Fake Images
    predicted_fake_imgs = discriminator(real_imgs)
    target_fake_imgs = torch.zeros(fake_imgs.size(0), 1)
    fake_imgs_loss = loss_function(predicted_fake_imgs, target_fake_imgs)
    fake_avg = torch.mean(predicted_fake_imgs).item()

    loss = real_imgs_loss + fake_imgs_loss
    loss.backward()
    disc_optimizer.step()

    return loss.item(), real_avg, fake_avg


# Running our whole Network:
def dcganModel(epochs, lr):
    torch.cuda.empty_cache()

    Gen_Loss = []
    Disc_Loss = []
    real_out = []
    fake_out = []
    # creating our optimizers for the Generator and Discriminator
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    s = 0
    for epoch in range(epochs):
        for real_img, _ in dataloader_train:
            s+=1

            disc_loss, real_imgs, fake_imgs = discriminator_train(real_img, disc_optimizer)
            gen_loss = generator_train(gen_optimizer)
            print("Batch{}".format(s))
            print(fake_imgs)
            Gen_Loss.append(gen_loss)
            Disc_Loss.append(disc_loss)
            real_out.append(real_imgs)
            fake_out.append(fake_imgs)
        # Plot last Fake output from Discriminator VS Real ones

    '''# Plotting Losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(Gen_Loss, label="G")
    plt.plot(Disc_Loss, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


    # Plotting Scores
    plt.plot(real_imgs, '-')
    plt.plot(fake_imgs, '-')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.legend(['Real', 'Fake'])
    plt.title('Scores');
'''
    return Gen_Loss, Disc_Loss, fake_out, real_out

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    # Training DataLoader with shuffling to ensure that data will be shuffled every epoch
    #  and num_workers because we will use multiple cores
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0)




    real_batch = next(iter(dataloader_train))
    Gen_Loss, Disc_Loss, fake_out, real_out = dcganModel(1, 0.2)
    print(fake_out)
    '''plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0], padding=5, normalize=True).cpu(), (1, 2, 0)))
    plt.show()
    '''
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(fake_out[0], (1, 2, 0)))
    plt.show()
'''
    # Plot the fake images from the last epoch
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(fake_out[-1], (1, 2, 0)))
    plt.show()
    # Plotting Losses
    plt.plot(Disc_Loss, '-')
    plt.plot(Gen_Loss, '-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Discriminator', 'Generator'])
    plt.title('Losses')'''

#    print(enumerate(dataset_train))