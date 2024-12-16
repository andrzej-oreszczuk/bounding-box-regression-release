from torch import nn, utils, optim, Tensor, cuda, squeeze
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



def calculate_label_from_crop(output, crop):

    left = crop[0] - crop[2] * output[2]
    right = crop[0] + crop[2] * output[3]
    top = crop[1] - crop[3] * output[0]
    bottom = crop[1] + crop[3] * output[1]

    x = (right + left) / 2
    y = (top + bottom) / 2
    w = (right - left)
    h = (bottom - top)

    return np.array([x, y, w, h])



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
            case "save_labels_directory":
                save_dir = option[1]
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

    model = light_model.model

    # dataloader
    dataset = images_with_locations(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    # create label files
    model.eval()

    for batch, data in enumerate(dataloader):
        print("location", batch, "/", len(dataloader))
        batch_images = data['image'].to(computational_device)
        batch_crop = data['crop']
        image_name = data['img_name'][0]

        batch_outputs = model(batch_images)
        output = Tensor.numpy(squeeze(batch_outputs), force=True)

        fl = open(save_dir + "/" + image_name.split(".jpg")[0] + "_output.txt", "a")
        line = "0 " + str(output[0]) + " " + str(output[1]) + " " + str(output[2]) + " " + str(output[3]) + "\n"
        fl.write(line)
        fl.close()

        crop = Tensor.numpy(squeeze(batch_crop), force=True)
        fl = open(save_dir + "/" + image_name.split(".jpg")[0] + "_crop.txt", "a")
        line = "0 " + str(crop[0]) + " " + str(crop[1]) + " " + str(crop[2]) + " " + str(crop[3]) + "\n"
        fl.write(line)
        fl.close()

        label = calculate_label_from_crop(output, crop)

        fl = open(save_dir + "/" + image_name.split(".jpg")[0] + ".txt", "a")
        line = "0 " + str(label[0]) + " " + str(label[1]) + " " + str(label[2]) + " " + str(label[3]) + "\n"
        fl.write(line)
        fl.close()





if __name__ == "__main__":
    main()
