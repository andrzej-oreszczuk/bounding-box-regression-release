from torch import nn, utils, optim, Tensor, cuda, squeeze
import os
import sys
import copy
from PIL import Image

import lightning as L

from torch.utils.data import Dataset, DataLoader

import numpy as np

from dataset_class_rotating import rotated_object

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

    locations = os.listdir(data_dir + "/locations_of_objects")

    dmax = (xmax ** 2 + ymax ** 2) ** (1 / 2)

    # load model
    light_model = bbr.load_from_checkpoint(model_path, bbr_model=bbr_model(computational_device, mode=1, mode_2=0))

    model = light_model.model

    # detect best angle
    model.eval()

    for l in locations:
        model(rotated_object(data_dir, data_dir + "/locations_of_objects/" + l, xmax, ymax, 0)["image"], rotated_object(data_dir, data_dir + "/locations_of_objects/" + l, xmax, ymax, 0)["location"])
        id = rotated_object(data_dir, data_dir + "/locations_of_objects/" + l, xmax, ymax, 0, mode=1)["id"]
        img_name = rotated_object(data_dir, data_dir + "/locations_of_objects/" + l, xmax, ymax, 0)["img_name"]
        filename = data_dir + "/cropped/" + img_name.split(".jpg")[0] + "_" + id + "_attention.jpg"
        os.rename('attention_map.png', filename)










if __name__ == "__main__":
    main()
