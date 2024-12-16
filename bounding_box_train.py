from torch import nn, utils, optim, Tensor, cuda
import os
import sys
import copy
from PIL import Image

import lightning as L

from torch.utils.data import Dataset, DataLoader

import numpy as np

from dataset_class import images_with_locations

from init_model import bbr_model_combined, bbr_model
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
    resume_training = 0
    model_type = "simple"
    model_mode = 21
    augmentation_flip_rotate = 0
    augmentation_dynamic_generation = 0
    augmentation_square = 0

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
            case "model_type":
                model_type = option[1]
            case "model_mode":
                model_mode = int(option[1])
            case "augmentation_dynamic_generation":
                augmentation_dynamic_generation = int(option[1])
            case "augmentation_flip_rotate":
                augmentation_flip_rotate = int(option[1])
            case "augmentation_square":
                augmentation_square = int(option[1])
            case _:
                print("Unknown config parameter")

    # load model
    match model_type:
        case "simple":
            if model_path != "##":
                light_model = bbr.load_from_checkpoint(model_path, bbr_model=bbr_model(computational_device, mode=model_mode))
            else:
                light_model = bbr(bbr_model(computational_device, mode=model_mode))

        case "combined":
            if model_path != "##":
                light_model = bbr.load_from_checkpoint(model_path, bbr_model=bbr_model_combined(computational_device))
            else:
                light_model = bbr(bbr_model_combined(computational_device))
        case _:
            print("Unknown model type")



    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor='valid_loss', filename='point-{epoch:02d}-{valid_loss:.6f}-{train_loss:.6f}-aug'+str(augmentation_flip_rotate)+str(augmentation_square), save_last=True, save_top_k=2)

    trainer = L.Trainer(limit_train_batches=1, log_every_n_steps=12, max_epochs=epochs, default_root_dir=model_save_path, callbacks=[checkpoint_callback], check_val_every_n_epoch=1)


    # dataloader

    trainset = images_with_locations(data_dir + "/train", augmentation_dynamic_generation=augmentation_dynamic_generation, augmentation_fr=augmentation_flip_rotate, augmentation_square=augmentation_square)
    trainloader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=18)

    validset = images_with_locations(data_dir + "/valid", augmentation_square=augmentation_square)
    validloader = DataLoader(validset, batch_size=batch, shuffle=False, num_workers=18)

    testset = images_with_locations(data_dir + "/test", augmentation_square=augmentation_square)
    testloader = DataLoader(testset, batch_size=batch, shuffle=False, num_workers=4)

    # training and validation
    if resume_training != 0:
        trainer.fit(light_model, trainloader, validloader, ckpt_path=model_path)
    else:
        trainer.fit(light_model, trainloader, validloader)

    # testing
    best_path = checkpoint_callback.best_model_path
    match model_type:
        case "simple":
            light_model = bbr.load_from_checkpoint(best_path, bbr_model=bbr_model(computational_device, mode=model_mode))

        case "combined":
            light_model = bbr.load_from_checkpoint(best_path, bbr_model=bbr_model_combined(computational_device))

    trainer.test(light_model, dataloaders=testloader)



if __name__ == "__main__":
    main()
