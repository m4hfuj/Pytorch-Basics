import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# hyper parameters
epochs = 4
batch_size = 4
learning_rate = 0.001

# CIFAR10
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size  , shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img/2  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

dataiter = iter(train_loader)
images, labels = dataiter.__next__()

# imshow(torchvision.utils.make_grid(images))

conv1 = nn.Conv2d(3, 6, 5)
pool = nn.MaxPool2d(2, 2)
conv2 = nn.Conv2d(6, 16, 5)
print('shape init                : ', images.shape)

x = conv1(images)
print('shape after 1st con layer : ', x.shape)

x = pool(x)
print('shape after 1st pool      : ', x.shape)

x = conv2(x)
print('shape after 2nd con layer : ', x.shape)

x = pool(x)
print('shape after 2nd pool      : ', x.shape)