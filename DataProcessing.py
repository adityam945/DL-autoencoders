import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, STL10
import os

import numpy as np

# Dataset load
def loader(train, test):
    train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True, num_workers=0)
    return train_loader, test_loader


def load_data_mnist():
    data_transform = transforms.Compose([
            transforms.ToTensor()
    ])
    train = MNIST(root="./data", train=True, transform=data_transform, download=True)
    test = MNIST(root="./data", train=False, transform=data_transform, download=True)
    print (np.array(train).shape, np.array(test).shape)

    return loader(train, test)


def load_data_fashion_mnist():
    data_transform = transforms.Compose([
            transforms.ToTensor()
    ])
    train = FashionMNIST(root="./data", train=True, transform=data_transform, download=True)
    test = FashionMNIST(root="./data", train=False, transform=data_transform, download=True)
    print (np.array(train).shape, np.array(test).shape)
    return loader(train, test)


def load_data_cifar():
    data_transform = transforms.Compose([
            transforms.ToTensor()
    ])
    train = CIFAR10(root="./data", train=True, transform=data_transform, download=True)
    test = CIFAR10(root="./data", train=False, transform=data_transform, download=True)
    print (np.array(train).shape, np.array(test).shape)
    print (np.array(train).shape, np.array(test).shape)

    return loader(train, test)


def load_data_stl():
    data_transform = transforms.Compose([
            transforms.ToTensor()
    ])
    train = STL10(root="./data", split="unlabeled", transform=data_transform, download=True)
    test = STL10(root="./data", split="test", transform=data_transform, download=True)
    print (np.array(train).shape, np.array(test).shape)

    return loader(train, test)


# image saving
def save_image_for_epoch(recon_batch, model_name, epoch):
    save_image(recon_batch, f"{model_name}/reconstruction_epoch_{str(epoch)}.png", nrow=8)


def check_create_path(model_name):
    if not os.path.exists(model_name):
        os.mkdir(model_name)