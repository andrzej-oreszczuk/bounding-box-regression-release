from torch import nn, utils, optim, Tensor, cuda
import os
import sys
import copy
from PIL import Image

import lightning as L

from torch.utils.data import Dataset, DataLoader

import numpy as np

from dataset_class import images_with_locations

from init_model import bbr_model
from lightning_module import bbr






def main():

    if cuda.is_available():
        computational_device = 'cuda'
    else:
        computational_device = 'cpu'

    # config parsing
    config_file = sys.argv[1]

    fl = open(config_file)
    config_data = fl.readlines()
    fl.close()

    batch = 0
    model_path = ""
    model_save_path = ""
    data_dir = ""
    epochs = 0
    xmax = 160
    ymax = 160
    resume_training = 0

    for line in config_data:
        option = line.split()
        match option[0]:
            case "batch":
                batch = int(option[1])
            case "model_path":
                model_path = option[1]
            case "model_save_path":
                model_save_path = option[1]
            case "data_dir":
                data_dir = option[1]
            case "epochs":
                epochs = int(option[1])
            case "resume_training":
                resume_training = int(option[1])
            case "labels_xmax":
                xmax = int(option[1])
            case "labels_ymax":
                ymax = int(option[1])

            case _:
                print("Unknown config parameter")

    # load model
    if model_path != "##":
        light_model = bbr.load_from_checkpoint(model_path, bbr_model=bbr_model(computational_device))
    else:
        light_model = bbr(bbr_model(computational_device))

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor='valid_loss', filename='point-{epoch:02d}-{valid_loss:.6f}-{train_loss:.6f}', save_top_k=2)

    trainer = L.Trainer(limit_train_batches=1.0, log_every_n_steps=12, max_epochs=epochs, default_root_dir=model_save_path, callbacks=[checkpoint_callback])


    # dataloader
    trainset = images_with_locations(data_dir + "/train", xmax, ymax)
    trainloader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=12)

    validset = images_with_locations(data_dir + "/valid", xmax, ymax)
    validloader = DataLoader(validset, batch_size=batch, shuffle=False, num_workers=12)

    testset = images_with_locations(data_dir + "/test", xmax, ymax)
    testloader = DataLoader(testset, batch_size=batch, shuffle=False, num_workers=12)

    # loader test
    '''
    for i_batch, sample_batched in enumerate(trainloader):
        print(i_batch, sample_batched['img_name'][0],sample_batched['location'][0], sample_batched['mask'][0,:])
    '''

    # training and validation
    if resume_training != 0:
        trainer.fit(light_model, trainloader, validloader, ckpt_path=model_path)
    else:
        trainer.fit(light_model, trainloader, validloader)

    # testing
    best_path = checkpoint_callback.best_model_path
    light_model = bbr.load_from_checkpoint(best_path, bbr_model=bbr_model(computational_device))
    trainer.test(light_model, dataloaders=testloader)



if __name__ == "__main__":
    main()
