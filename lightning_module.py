from torch import nn, utils, optim, Tensor, cuda

import lightning as L

class bbr(L.LightningModule):
    def __init__(self, bbr_model):
        super().__init__()
        self.model = bbr_model

    def training_step(self, batch, batch_idx):
        batch_images = batch['image']
        batch_labels = batch['label']
        batch_location = batch['location']

        output = self.model(batch_images, batch_location)

        loss = nn.functional.l1_loss(output, batch_labels)

        self.log("train_loss", loss, batch_size=batch_labels.size(dim=0), on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        batch_images = batch['image']
        batch_labels = batch['label']
        batch_location = batch['location']

        output = self.model(batch_images, batch_location)

        loss = nn.functional.mse_loss(output, batch_labels)

        self.log("test_loss", loss, batch_size=batch_labels.size(dim=0), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        batch_images = batch['image']
        batch_labels = batch['label']
        batch_location = batch['location']
        output = self.model(batch_images, batch_location)

        loss = nn.functional.mse_loss(output, batch_labels)

        self.log("valid_loss", loss, batch_size=batch_labels.size(dim=0), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-10)
        return optimizer


