import os
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms

def load_mnist_dataset(mode:str, is_valid:bool=False, file_locate:str=os.getcwd()):
    datasets= {}
    dataset = None
    if mode == 'train':
        dataset = MNIST(file_locate, train=True, download=True, transform=transforms.ToTensor())
    else:
        dataset = MNIST(file_locate, download=True, transform=transforms.ToTensor())
    if is_valid:
        train_size = int(0.8 * len(dataset))
        valid_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, valid_size])
        datasets['train'] = train_dataset
        datasets['val'] = val_dataset
        return datasets
    return dataset