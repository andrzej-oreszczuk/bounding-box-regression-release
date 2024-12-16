
import torch
import torchvision
import os
import sys
import math
import random
import cv2
from PIL import Image, ImageDraw

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np

def calculate_target(label, crop):
    l = (crop[0] - (label[0] - label[2]/2))/crop[2]
    r = ((label[0] + label[2]/2) - crop[0])/crop[2]
    t = (crop[1] - (label[1] - label[3] / 2))/crop[3]
    b = ((label[1] + label[3] / 2) - crop[1])/crop[3]
    return np.array([t, b, l, r])


class images_with_locations(Dataset):

    def __init__(self, root_dir, augmentation_dynamic_generation=0, augmentation_fr=0, augmentation_square=0, mode=0):
        """
        Arguments:
            csv_file (string): Path to the csv file with locationss.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mode = mode
        self.objects_dir = root_dir + "/approximate_labels"
        self.root_dir = root_dir
        self.augmentation_dynamic_generation = augmentation_dynamic_generation
        self.augmentation_square = augmentation_square
        self.augmentation_fr = augmentation_fr


    def __len__(self):
        return len(os.listdir(self.objects_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        dir_list = os.listdir(self.objects_dir)

        fl = open(self.objects_dir + "/" + dir_list[idx])
        file_data = fl.readlines()
        fl.close()
        #print(self.locations_dir + "/" + dir_list[idx])

        first_line = file_data[0].split()

        label_number = first_line[-1]
        image_name = file_data[0].rstrip()[:-(len(label_number)+1)]
        image_file = self.root_dir + "/images/" + image_name


        labels_file = self.root_dir + "/labels/" + image_name[:-4] + ".txt"

        # extract label
        fl = open(labels_file)
        labels_data = fl.readlines()
        fl.close()

        label_str = labels_data[int(label_number) - 1]
        edges = label_str.split()
        x_true = float(edges[1])
        y_true = float(edges[2])
        w_true = float(edges[3])
        h_true = float(edges[4])

        label = np.array([x_true, y_true, w_true, h_true])

        # load image
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        yi, xi = image.shape[:2]

        # extract approximate label
        location_str = file_data[1]
        _xywh = location_str.split()
        x = float(_xywh[1])
        y = float(_xywh[2])
        w = float(_xywh[3])
        h = float(_xywh[4])

        x_augmented = float(_xywh[1])
        y_augmented = float(_xywh[2])
        w_augmented = float(_xywh[3])
        h_augmented = float(_xywh[4])



        # generating approximate label

        if self.augmentation_dynamic_generation > 0:
            x = x_true
            y = y_true
            w = w_true
            h = h_true
            # generated label
            xl = x
            yl = y
            wl = w
            hl = h

            # generate top of the label
            dt = 1
            s = 1
            while (y - h / 2 - dt < 0.001):
                s *= 0.95
                dt = (0.1 + random.random() / 2) * s * h
            yl -= dt / 2
            hl += dt
            # generate bottom of the label:
            dt = 1
            s = 1
            while (y + h / 2 + dt > 0.999):
                s *= 0.95
                dt = (0.1 + random.random() / 2) * s * h
            yl += dt / 2
            hl += dt
            # generate left bound of the label:
            dt = 1
            s = 1
            while (x - w / 2 - dt < 0.001):
                s *= 0.95
                dt = (0.1 + random.random() / 2) * s * w
            xl -= dt / 2
            wl += dt
            # generate right bound of the label:
            dt = 1
            s = 1
            while (x + w / 2 + dt > 0.999):
                s *= 0.95
                dt = (0.1 + random.random() / 2) * s * w
            xl += dt / 2
            wl += dt

            x_augmented = xl
            y_augmented = yl
            w_augmented = wl
            h_augmented = hl

        # augmentation squaring
        if self.augmentation_square == 1:
            if w > h:
                dt = (w - h)
                s = 1
                while (y - h / 2 - dt / 2 < 0.001 or y + h / 2 + dt / 2 > 0.999):
                    dt = s * (w - h)
                    s *= 0.9
                h_augmented += dt
            else:
                dt = (h - w)
                s = 1
                while (x - w / 2 - dt / 2 < 0.001 or x + w / 2 + dt / 2 > 0.999):
                    dt = s * (h - w)
                    s *= 0.9
                w_augmented += dt

        # calculate target

        crop = np.array([x_augmented, y_augmented, w_augmented, h_augmented])
        target = calculate_target(label, crop)

        # crop object
        x_crop = round(x_augmented * xi)
        y_crop = round(y_augmented * yi)
        w_crop = round(w_augmented * xi)
        h_crop = round(h_augmented * yi)

        if self.mode != 0:
            x1, y1 = int(xi * x_true - 3), int(yi * y_true - 3)
            x2, y2 = int(xi * x_true + 3), int(yi * y_true + 3)

            color = (0, 0, 0)  # Black in BGR format

            image = cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=1)
        

        # crop and resize
        x1 = x_crop - round(w_crop / 2)
        y1 = y_crop - round(h_crop / 2)
        x2 = x_crop + round(w_crop / 2)
        y2 = y_crop + round(h_crop / 2)
        
        cropped = image[y1:y2, x1:x2].copy()

        cropped = cv2.resize(cropped, (218, 218))


        # augmentation mirroring and rotations
        rotation = 0
        flip_horizontal = 0
        flip_vertical = 0
        if self.augmentation_fr == 1:
            # rotation
            rotation = random.randrange(1, 3) - 1
            xt = 0
            if rotation == 1:
                cropped = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)

                xt = target[0]
                target[0] = target[3]
                target[3] = target[1]
                target[1] = target[2]
                target[2] = xt

            # random flip
            flip_vertical = random.randrange(1, 3) - 1
            if flip_vertical == 1:

                cropped = cv2.flip(cropped, 0)
                xt = target[0]
                target[0] = target[1]
                target[1] = xt

            flip_horizontal = random.randrange(1, 3) - 1
            if flip_horizontal == 1:
                cropped = cv2.flip(cropped, 1)
                xt = target[2]
                target[2] = target[3]
                target[3] = xt



        if self.mode != 0:
            if not os.path.exists(self.root_dir + "/cropped_test/"):
                os.mkdir(self.root_dir + "/cropped_test/")

            path = self.root_dir + "/cropped_test/" + first_line[0] + first_line[1] + ".jpg"
            cv2.imwrite(path, cropped)

        transform = transforms.Compose([transforms.ToTensor()])

        #create sample

        image_tensor = transform(cropped)

        augmentations_tensor = torch.tensor(np.array([rotation, flip_vertical, flip_horizontal]), dtype=torch.float32)

        crop_tensor = torch.tensor(crop, dtype=torch.float32)

        target_tensor = torch.tensor(target, dtype=torch.float32)

        label_tensor = torch.tensor(label, dtype=torch.float32)

        image_size_tensor = torch.tensor([xi, yi], dtype=torch.float32)

        sample = {"image": image_tensor, "crop": crop_tensor,
                  "label": label_tensor, "img_name": image_file.split('/')[-1], "image_size": image_size_tensor,
                  "target": target_tensor, "augmentations": augmentations_tensor}

        return sample


def main():
    # dataloader
    random.seed()
    data_dir = sys.argv[1]
    testset = images_with_locations(data_dir, augmentation_dynamic_generation=0, augmentation_fr=0, mode=1)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # loader test
    for i_batch, sample_batched in enumerate(testloader):
        print(i_batch, sample_batched['img_name'][0], sample_batched['crop'][0])



if __name__ == "__main__":
    main()
