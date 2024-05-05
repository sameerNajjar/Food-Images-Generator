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
import os

# this line removes a warning that tensorboard was printing
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def filter_dataset(dataset):
    filtered_data = []
    for i in range(len(dataset)):
        try:
            img, _ = dataset[i]
            filtered_data.append((img, _))
        except Exception as e:
            print(f"Error loading image at index {i}: {e}")
            continue
    return filtered_data


def add_gaussian_noise(x):
    return x + torch.randn_like(x) * 0.1


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cuda':
        torch.multiprocessing.freeze_support()
    lr = 0.0001
    batch_size = 64
    image_size = 64
    critic_features = 64
    gen_features = 64
    img_channels = 3
    noise_dom = 100
    gen_itr = 5
    lambda_gp = 10
    epochs = 200

    transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # transforms.Lambda(add_gaussian_noise)
    ])
    # dataset_path = "clothes"
    dataset_path = "food-101/food-101/images"
    dataset = datasets.ImageFolder(root=dataset_path, transform=transforms)
    # dataset = filter_dataset(dataset)
    # dataset = datasets.MNIST(root="mnist/", transform=transforms, download=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=4,
    )

    gen = Generator(noise_dom, img_channels, gen_features).to(device)
    critic = Discriminator(img_channels, critic_features).to(device)
    initialize_weights(gen)
    initialize_weights(critic)
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))
    writer_real = SummaryWriter(f"runs/real")
    writer_fake = SummaryWriter(f"runs/fake")
    writer_gen_loss = SummaryWriter(f"runs/gen_loss")
    writer_critic_loss = SummaryWriter(f"runs/critic_loss")
    step = 0
    for epoch in range(epochs):
        for i, (images, _) in enumerate(tqdm(loader)):
            real = images.to(device)
            noise = torch.randn(batch_size, noise_dom, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic += lambda_gp * gp
            opt_critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()
            if i % gen_itr == 0:
                gen_fake = critic(fake).reshape(-1)
                loss_gen = -torch.mean(gen_fake)
                opt_gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()

            if i % 500 == 0 and i > 0:
                print(f"Epoch [{epoch}/{epochs}] Batch {i}/{len(loader)} ")
                writer_critic_loss.add_scalar('Critic Loss', loss_critic.item(), global_step=step)
                writer_gen_loss.add_scalar('Generator Loss', loss_gen.item(), global_step=step)
                with torch.no_grad():
                    fake = gen(noise)
                    img_grid_real = torchvision.utils.make_grid(real[:32],normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32],normalize=True)
                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1
        torch.save(gen.state_dict(), 'pre-trained-models/Generator.pth')
        torch.save(critic.state_dict(), 'pre-trained-models/Discriminator.pth')

