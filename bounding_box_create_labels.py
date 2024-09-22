from torch import nn, utils, optim, Tensor, cuda, squeeze
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

    batch = 1
    model_path = ""
    data_dir = ""
    xmax = 160
    ymax = 160

    for line in config_data:
        option = line.split()
        match option[0]:
            case "model_path":
                model_path = option[1]
            case "data_dir":
                data_dir = option[1]
            case "save_labels_directory":
                save_dir = option[1]
            case "labels_xmax":
                xmax = int(option[1])
            case "labels_ymax":
                ymax = int(option[1])
            case _:
                print("Unknown config parameter")

    # load model
    light_model = bbr.load_from_checkpoint(model_path, bbr_model=bbr_model(computational_device))

    model = light_model.model

    # dataloader
    dataset = images_with_locations(data_dir, xmax, ymax)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    # create label files
    model.eval()

    for batch, data in enumerate(dataloader):
        print("location", batch, "/", len(dataloader))
        batch_images = data['image'].to(computational_device)

        image_name = data['img_name'][0]
        batch_location = data['location'].to(computational_device)

        batch_outputs = model(batch_images, batch_location)
        output = Tensor.numpy(squeeze(batch_outputs), force=True)
        fl = open(save_dir + "/" + image_name.split(".jpg")[0] + ".txt", "a")
        line = "0 " + str(output[0]) + " " + str(output[1]) + " " + str(output[2]) + " " + str(output[3]) + "\n"
        fl.write(line)
        fl.close()





if __name__ == "__main__":
    main()
