import torch
import pytorch_lightning as pl
from models import LinearModel
from lightning import LitLinear
from dataset import load_mnist_dataset
from torch.utils.data import Dataloader


if __name__ == "__main__":

    AVAIL_GPU = min(1, torch.cuda.device_count())
    BATCH_SIZE = 32
    EPOCH = 20
    IS_VALID = True
    pl.seed_everything(8)
    
    datasets = load_mnist_dataset('test', IS_VALID)
    test_loader = Dataloader(datasets)
    model = LitLinear(LinearModel())

    #model load 하는 부분 추가하자
    trainer = pl.Trainer(gpus=AVAIL_GPU)
    trainer.test(model, test_loader)
