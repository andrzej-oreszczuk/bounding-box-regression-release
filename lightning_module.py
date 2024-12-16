from torch import nn, utils, optim, Tensor, cuda
import torchvision

import lightning as L

class bbr(L.LightningModule):
    def __init__(self, bbr_model):
        super().__init__()
        self.model = bbr_model

    def training_step(self, batch, batch_idx):
        batch_images = batch['image']
        batch_target = batch['target']

        output = self.model(batch_images)

        loss = nn.functional.mse_loss(output, batch_target)

        self.log("train_loss", loss, batch_size=batch_target.size(dim=0), prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        batch_images = batch['image']
        batch_target = batch['target']

        output = self.model(batch_images)

        loss = nn.functional.mse_loss(output, batch_target)
        loss_mae = nn.functional.l1_loss(output, batch_target)

        batch_image_size = batch['image_size'].detach().cpu().numpy()
        batch_crop = batch['crop'].detach().cpu().numpy()
        np_target = batch['target'].detach().cpu().numpy()
        np_output = output.detach().cpu().numpy()


        loss_mae_pixel_sum = 0

        for i in range(batch_target.size(dim=0)):
            dt = np_target[i][0] - np_output[i][0]
            db = np_target[i][1] - np_output[i][1]
            dl = np_target[i][2] - np_output[i][2]
            dr = np_target[i][3] - np_output[i][3]

            #crop dimensions:
            ch = batch_crop[i][3]
            cw = batch_crop[i][2]

            #image dimensions:
            ih = batch_image_size[i][0]
            iw = batch_image_size[i][1]

            loss_mae_pixel_sum += (abs(dt*ch*ih) + abs(db*ch*ih) + abs(dl*cw*iw) + abs(dr*cw*iw))/4

            fl = open("log_loss.txt", "a")
            line = str(dt) + " " + str(db) + " " + str(dl) + " " + str(dr) + " " + str(ch) + " " + str(cw) + " " + str(ih) + " " + str(iw) + " " + str(np_target[i][0]) + " " + str(np_target[i][1]) + " " + str(np_target[i][2]) + " " + str(np_target[i][3]) + "\n"
            fl.write(line)
            fl.close()

        self.log("test_loss", loss, batch_size=batch_target.size(dim=0), on_step=False, on_epoch=True)
        self.log("test_loss_mae", loss_mae, batch_size=batch_target.size(dim=0), on_step=False, on_epoch=True)
        self.log("test_loss_mae_pixel", loss_mae_pixel_sum/batch_target.size(dim=0), batch_size=batch_target.size(dim=0), on_step=False, on_epoch=True)


    def validation_step(self, batch, batch_idx):
        batch_images = batch['image']
        batch_target = batch['target']

        output = self.model(batch_images)

        loss = nn.functional.mse_loss(output, batch_target)

        self.log("valid_loss", loss, batch_size=batch_target.size(dim=0), prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-15)
        return optimizer


