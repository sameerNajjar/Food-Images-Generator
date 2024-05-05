import torch
import torch.nn as nn
import torch.autograd as autograd


class Discriminator(nn.Module):
    def __init__(self, channels, features):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels, features, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(features*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(features * 8, 1, kernel_size=4, stride=2, padding=0))
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, noise_dim, img_channels, features):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, features * 8, 4, 1, 0),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.ConvTranspose2d(features, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)


def gradient_penalty(critic, real, fake, device):
    size, channels, h, w = real.shape
    alpha = torch.rand((size, 1, 1, 1)).repeat(1, channels, h, w).to(device)
    interpolates = real * alpha + fake * (1 - alpha)
    disc_interpolates = critic(interpolates)
    gradient = autograd.grad(
        inputs=interpolates,
        outputs=disc_interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gp = torch.mean((gradient.norm(2, dim=1) - 1) ** 2)
    return gp
