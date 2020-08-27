import numpy as np

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = torchvision.datasets.SVHN(
    root='./data', split='train', download=True, transform=transform_train)
testset = torchvision.datasets.SVHN(
    root='./data', split='test', download=True, transform=transform_test)

dataset_size = len(full_dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.2 * dataset_size))
np.random.seed(1)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=10)
