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

    dmax = (xmax**2 + ymax**2)**(1/2)


    # load model
    light_model = bbr.load_from_checkpoint(model_path, bbr_model=bbr_model(computational_device))

    model = light_model.model


    # detect best angle
    model.eval()

    for l in locations:

        best_angle = 0
        min_area = 1.0

        for a in range(360):
            created_label = model(rotated_object(data_dir, data_dir + "/locations_of_objects/"+l, dmax, dmax, a)["image"], rotated_object(data_dir, data_dir + "/locations_of_objects/"+l, dmax, dmax, a)["location"])
            created_label = created_label.squeeze()
            area = created_label[2]*created_label[3]
            if area < min_area:
                min_area = area
                best_angle = a
        rotated_object(data_dir, data_dir + "/locations_of_objects/" + l, dmax, dmax, best_angle, 1)









if __name__ == "__main__":
    main()
