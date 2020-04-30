import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_dataset(batch_size, data_size, train):
    t = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda img: img.flatten())])
    dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=t)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(range(data_size)))

    x, y = zip(*list(loader))
    return np.stack(x), np.stack(y)
