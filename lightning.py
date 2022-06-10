import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim

class LitLinear(pl.LightningModule):

    def __init__(self, linear):
        super().__init__()
        self.model = linear

    def training_step(self, batch, batch_idx):
        x,y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat,y)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat,y)
        self.log("Train loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def configure_optimizers(self):
        optm = optim.Adam(self.parameters(), lr=1e-3)
        return optm