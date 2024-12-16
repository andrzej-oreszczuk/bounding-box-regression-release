from torch import nn, utils, optim, Tensor, cuda, squeeze
import os
import sys
import matplotlib.pyplot as plt
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
    model_type = "simple"
    model_mode = 21

    for line in config_data:
        option = line.split()
        match option[0]:
            case "batch":
                batch = int(option[1])
            case "model_path":
                model_path = option[1]
            case "data_dir":
                data_dir = option[1]
            case "model_type":
                model_type = option[1]
            case "model_mode":
                model_mode = int(option[1])
            case _:
                print("Unknown config parameter")

    # load model
    match model_type:
        case "simple":
            if model_path != "##":
                light_model = bbr.load_from_checkpoint(model_path,
                                                       bbr_model=bbr_model(computational_device, mode=model_mode))
            else:
                light_model = bbr(bbr_model(computational_device, mode=model_mode))
        case _:
            print("Unknown model type")

    locations = os.listdir(data_dir + "/approximate_labels")

    model = light_model.model

    # detect best angle
    model.eval()

    for l in locations:

        best_angle = 0
        min_average_area = 1.0
        areas = []
        print(l)
        for a in range(360):
            rotated = rotated_object(data_dir, data_dir + "/approximate_labels/"+l, a)
            #print( rotated["location"], rotated["image_size"])
            created_prediction = model(rotated["image"].to(computational_device))
            prediction = Tensor.numpy(squeeze(created_prediction), force=True)


            w = prediction[2] + prediction[3]
            h = prediction[0] + prediction[1]

            area = w*h
            areas.append(area)
        for a in range(360):
            averaged_area = 0
            for i in range(21):
                if a-10+i < 360:
                    averaged_area += areas[a-10+i]
                else:
                    averaged_area += areas[a - 10 + i-360]
            averaged_area /= 21
            if averaged_area < min_average_area:
                best_angle = a
                min_average_area = averaged_area
        print(best_angle)


        #plt.plot(areas)
        #plt.ylabel(best_angle)
        #plt.show()

        rotated = rotated_object(data_dir, data_dir + "/approximate_labels/" + l, best_angle)
        # print( rotated["location"], rotated["image_size"])
        created_prediction = model(rotated["image"].to(computational_device))

        prediction = Tensor.numpy(squeeze(created_prediction), force=True)

        rotated_object(data_dir, data_dir + "/approximate_labels/" + l, best_angle, 2, prediction)











if __name__ == "__main__":
    main()
