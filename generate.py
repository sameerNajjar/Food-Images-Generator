import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import *
import matplotlib.pyplot as plt
import os


def generate(generator, noise_dim, num_images, device):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, noise_dim, 1, 1).to(device)
        generated_images = generator(noise).cpu()
    generator.train()
    return generated_images


def create_image_grid(images, num_rows=4, num_cols=8, normalize=True):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 16))
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            if index >= images.shape[0]:
                axes[i, j].axis('off')
                continue
            img = images[index]
            if normalize:
                img = (img.permute(1, 2, 0) + 1) / 2
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
    plt.setp(axes, xticks=[], yticks=[])
    plt.setp(fig.get_axes(), xlabel='', ylabel='')
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_channels = 3
    noise_dim = 100
    critic_features = 64
    gen_features = 64

    gen = Generator(noise_dim, img_channels, gen_features).to(device)
    critic = Discriminator(img_channels, critic_features).to(device)
    initialize_weights(gen)
    initialize_weights(critic)

    gen.load_state_dict(torch.load('pre-trained-models/Generator.pth'))
    num_images_to_generate = 32
    generated_images = generate(gen, noise_dim, num_images_to_generate, device)

    num_images_to_plot = min(len(generated_images), 32)
    image_grid = create_image_grid(generated_images[:num_images_to_plot])
    plt.show()
