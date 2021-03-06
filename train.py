from lightning import LitLinear
from models import LinearModel
from dataset import load_mnist_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

'''
Reference
- https://sdc-james.gitbook.io/onebook/9.6-6.-pytorch-lightning
- https://pytorch-lightning.readthedocs.io/en/latest/model/train_model_basic.html
'''


if __name__ == "__main__":

    #define Fields
    AVAIL_GPU = min(1, torch.cuda.device_count())
    BATCH_SIZE = 32
    EPOCH = 20
    IS_VALID = True
    pl.seed_everything(8)
    
    datasets = load_mnist_dataset('train', IS_VALID)
    train_dataloader = DataLoader(datasets['train'], batch_size=BATCH_SIZE, num_workers=4)
    val_datalodaer = None
    if IS_VALID:
        val_datalodaer = DataLoader(datasets['val'], batch_size=BATCH_SIZE, num_workers=4)


    model = LitLinear(LinearModel())
    optimizer = model.configure_optimizers()

    checkpoint_callback = ModelCheckpoint(
        filename='mnist-epoch{epoch:02d}-val_loss{val/loss:.2f}',
        verbose=True,
        save_last=True,
        monitor='val_acc', # 이거 찾아보고 다시 해야함
        mode='max'
    )

    trainer = pl.Trainer(gpus=AVAIL_GPU, max_epochs=EPOCH, progress_bar_refresh_rate=20, callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_datalodaer)