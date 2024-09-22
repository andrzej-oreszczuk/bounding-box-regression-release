
import torch
import torchvision
import os
import sys
import math
from PIL import Image, ImageDraw

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np



def rotated_object(root_dir, location_file, xmax_in, ymax_in, alphad, mode = 0):

    xmax = xmax_in + 20
    ymax = ymax_in + 20

    fl = open(location_file)
    file_data = fl.readlines()
    fl.close()

    first_line = file_data[0].split()

    img_name = root_dir + "/images/" + first_line[0]
    image = Image.open(img_name).convert('L')
    xi, yi = image.size

    alpha = alphad/360.0 * 2 * math.pi

    ws = math.sin(alpha) * yi + math.cos(alpha) * xi
    hs = math.cos(alpha) * yi + math.sin(alpha) * xi

    # location coordinates
    location_str = file_data[1]
    xy = location_str.split()
    location = np.array([float(xy[0]), float(xy[1])])
    x = float(xy[0]) * xi
    y = float(xy[1]) * yi

    #draw = ImageDraw.Draw(image)
    #draw.rectangle(xy=(x - 4, y - 4, x + 4, y + 4),
    #               fill=(0),
    #               outline=(255),
    #               width=5)

    xs = x + (ws-xi)/2
    ys = y + (hs-yi)/2

    xsr = (xs - ws / 2) * math.cos(-alpha) - (ys - hs / 2) * math.sin(-alpha) + ws/2
    ysr = (xs - ws / 2) * math.sin(-alpha) + (ys - hs / 2) * math.cos(-alpha) + hs/2

    # crop image
    image = image.rotate(alphad, expand=True)
    #draw2 = ImageDraw.Draw(image)
    #draw2.rectangle(xy=(xsr - 4, ysr - 4, xsr + 4, ysr + 4),
    #               fill=(0),
    #               outline=(255),
    #               width=5)
    #if mode != 0:
        #print(xi, yi, x, y, ws, hs, xs, ys, xsr, ysr)
        #image.save(root_dir + first_line[0] + xy[0] + "full.jpg")

    cropped = image.crop((xsr - xmax/2, ysr - ymax/2, xsr+xmax/2, ysr+ymax/2))

    cropped = cropped.resize((201, 201))

    if mode != 0:
        if not os.path.exists(root_dir + "/cropped/"):
            os.mkdir(root_dir + "/cropped/")
        cropped.save(root_dir + "/cropped/" + first_line[0] + first_line[1] + "angle=" + str(alphad) + ".jpg")

    transform = transforms.Compose([transforms.PILToTensor()])

    image_tensor = transform(cropped)/255

    location_tensor = torch.tensor([xsr, ysr], dtype=torch.float32)

    sample = {"image": image_tensor, "location": location_tensor, "img_name": img_name.split('/')[-1], "id": first_line[1]}

    return sample



def main():
    # dataloader
    data_dir = sys.argv[1]
    locations = os.listdir(data_dir + "/locations_of_objects")

    # loader test

    for l in locations:
        print(rotated_object(data_dir, data_dir + "/locations_of_objects/"+l , 491, 134, 0, 1)['img_name'])



if __name__ == "__main__":
    main()
