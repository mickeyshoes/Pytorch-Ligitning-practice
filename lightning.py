import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim

class LitLinear(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x,y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat,y)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat,y)
        self.log("Train loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def configure_optimizers(self):
        optm = optim.Adam(self.parameters(), lr=1e-3)
        return optm