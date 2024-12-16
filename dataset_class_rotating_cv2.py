
import torch
import torchvision
import os
import sys
import math
from tools import coordinates
from PIL import Image, ImageDraw

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np



def rotated_object(root_dir, location_file, alphad, mode = 0, prediction = [0,0,0,0]):

    fl = open(location_file)
    file_data = fl.readlines()
    fl.close()

    first_line = file_data[0].split()

    label_number = first_line[-1]
    image_name = file_data[0].rstrip()[:-(len(label_number) + 1)]
    image_file = root_dir + "/images/" + image_name

    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    yi, xi = image.shape[:2]

    #draw ground truth
    if mode == 2:
        # extract label
        labels_file = root_dir + "/labels/" + image_name[:-4] + ".txt"

        fl = open(labels_file)
        labels_data = fl.readlines()
        fl.close()
        label_str = labels_data[int(label_number) - 1]

        edges = label_str.split()
        x_true = float(edges[1])
        y_true = float(edges[2])
        w_true = float(edges[3])
        h_true = float(edges[4])

        l, t, r, b = coordinates(x_true, y_true, w_true, h_true, xi, yi)
        cv2.rectangle(image, (l, t), (r, b), 0, 3)

    alpha = alphad/360.0 * 2 * math.pi

    ws = abs(math.sin(alpha)) * yi + abs(math.cos(alpha)) * xi
    hs = abs(math.cos(alpha)) * yi + abs(math.sin(alpha)) * xi

    # location coordinates
    location_str = file_data[1]
    label = location_str.split()
    location = np.array([float(label[1]), float(label[2])])

    width = ((float(label[4]) * yi)**2 + (float(label[3])*xi)**2)**(1/2)
    height = width

    x = float(location[0]) * xi
    y = float(location[1]) * yi

    xs = x + (ws-xi)/2
    ys = y + (hs-yi)/2

    xsr = (xs - ws / 2) * math.cos(-alpha) - (ys - hs / 2) * math.sin(-alpha) + ws/2
    ysr = (xs - ws / 2) * math.sin(-alpha) + (ys - hs / 2) * math.cos(-alpha) + hs/2

    # crop image
    image = image.rotate(alphad, expand=True)

    cropped = image.crop((xsr - width/2, ysr - height/2, xsr+width/2, ysr+height/2))

    cropped = cropped.resize((218, 218))

    if mode == 1:
        if not os.path.exists(root_dir + "/cropped/"):
            os.mkdir(root_dir + "/cropped/")

        cropped.save(root_dir + "/cropped/" + first_line[0] + first_line[1] + "angle=" + str(alphad) + ".jpg")


    # draw prediction
    if mode == 2:
        if not os.path.exists(root_dir + "/cropped_pred/"):
            os.mkdir(root_dir + "/cropped_pred/")

        tp = 109 - prediction[0] * 109
        bp = 109 + prediction[1] * 109
        lp = 109 - prediction[2] * 109
        rp = 109 + prediction[3] * 109
        img2 = ImageDraw.Draw(cropped)
        img2.rectangle([(lp, tp), (rp, bp)], width=1)

        cropped.save(root_dir + "/cropped_pred/" + first_line[0] + first_line[1] + "angle=" + str(alphad) + ".jpg")


    transform = transforms.Compose([transforms.PILToTensor()])

    image_tensor = (transform(cropped)/256.0).unsqueeze(0)

    crop_size_tensor = torch.tensor([width, height], dtype=torch.float32)

    image_size_tensor = torch.tensor([xi, yi], dtype=torch.float32)

    sample = {"image": image_tensor, "crop_size": crop_size_tensor, "img_name": image_name, "id": first_line[-1], "image_size": image_size_tensor}



    return sample



def main():
    # dataloader
    data_dir = sys.argv[1]
    locations = os.listdir(data_dir + "/approximate_labels")

    # loader test

    for l in locations:
        print(rotated_object(data_dir, data_dir + "/approximate_labels/"+l, 290, 1)['img_name'])



if __name__ == "__main__":
    main()
