from unicodedata import name
from lightning import LitLinear
from models import LinearModel
from albumentations.pytorch.transforms import ToTensorV2
import os
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torch

'''
Reference
- https://sdc-james.gitbook.io/onebook/9.6-6.-pytorch-lightning
- https://pytorch-lightning.readthedocs.io/en/latest/model/train_model_basic.html
'''


def load_dataset(is_valid:bool):
    datasets= {}
    dataset = MNIST(os.getcwd(), download=False, transform=ToTensorV2())
    if is_valid:
        train_dataset, val_dataset = random_split(dataset, [50000,5000])
        datasets['train'] = train_dataset
        datasets['val'] = val_dataset
        return datasets
    return dataset

if name == "__main__":

    #define Fields
    AVAIL_GPU = min(1, torch.cuda.device_count())
    BATCH_SIZE = 32
    IS_VALID = True
    pl.seed_everything(8)
    
    datasets = load_dataset(IS_VALID)
    train_dataloader = DataLoader(datasets['train'], batch_size=BATCH_SIZE)
    if IS_VALID: 
        val_datalodaer = DataLoader(datasets['val'], batch_size=BATCH_SIZE)


    model = LitLinear(LinearModel())
    optimizer = model.configure_optimizers()
    trainer = pl.Trainer()
    trainer.fit(model=model, train_dataloaders=train_dataloader)
