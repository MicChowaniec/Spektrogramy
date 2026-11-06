"""All weights are initialized from scratch; no pretrained downloads."""
import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(128, classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class VGG(nn.Module):
    """VGG11 convolutional configuration with a compact audio classifier head."""
    def __init__(self, classes):
        super().__init__()
        layers, channels = [], 1
        for value in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]:
            if value == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers.extend([nn.Conv2d(channels, value, 3, padding=1), nn.ReLU()])
                channels = value
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(512, classes))

    def forward(self, x):
        return self.head(self.features(x))


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    """Conditional generator: class labels make generated augmentation usable."""
    def __init__(self, classes, latent=64):
        super().__init__()
        self.latent = latent
        self.embedding = nn.Embedding(classes, latent)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * latent, 128, 4), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Tanh())

    def forward(self, z, labels):
        return self.net(torch.cat([z, self.embedding(labels)], 1)[:, :, None, None])


class Discriminator(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.embedding = nn.Embedding(classes, 64 * 64)
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 8), nn.Flatten())

    def forward(self, x, labels):
        condition = self.embedding(labels).reshape(-1, 1, 64, 64)
        return self.net(torch.cat([x, condition], 1)).flatten()
