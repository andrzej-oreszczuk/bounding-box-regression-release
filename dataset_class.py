
import torch
import torchvision
import os
import sys
import math
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np


class images_with_locations(Dataset):

    def __init__(self, root_dir, xmax, ymax, mode=0):
        """
        Arguments:
            csv_file (string): Path to the csv file with locationss.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mode = mode
        self.ymax = ymax + 20
        self.xmax = xmax + 20
        self.locations_dir = root_dir + "/locations_of_objects"
        self.root_dir = root_dir

    def __len__(self):
        return len(os.listdir(self.locations_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dir_list = os.listdir(self.locations_dir)

        fl = open(self.locations_dir + "/" + dir_list[idx])
        file_data = fl.readlines()
        fl.close()
        #print(self.locations_dir + "/" + dir_list[idx])

        first_line = file_data[0].split()

        img_name = self.root_dir + "/images/" + first_line[0]
        label_number = first_line[1]
        image = Image.open(img_name).convert('L')
        xi, yi = image.size

        # location coordinates
        location_str = file_data[1]
        xy = location_str.split()
        location = np.array([float(xy[0]), float(xy[1])])
        x = float(xy[0]) * xi
        y = float(xy[1]) * yi

        # crop image
        padding = Image.new(image.mode, (xi + self.xmax, yi + self.ymax), (0))
        padding.paste(image, (round(self.xmax / 2), round(self.ymax / 2)))
        cropped = padding.crop((x, y, x+self.xmax, y+self.ymax))

        cropped = cropped.resize((201, 201))

        if self.mode != 0:
            cropped.save(self.root_dir + first_line[0] + xy[0] + ".jpg")


        transform = transforms.Compose([transforms.PILToTensor()])

        image_tensor = transform(cropped)/255





        location_tensor = torch.tensor(location, dtype=torch.float32)

        labels_file = self.root_dir + "/labels/" + first_line[0].rstrip()[:-4] + ".txt"

        fl = open(labels_file)
        labels_data = fl.readlines()
        fl.close()

        label_str = labels_data[int(label_number) - 1]
        edges = label_str.split()
        label = np.array([float(edges[1]), float(edges[2]), float(edges[3]), float(edges[4])])


        label_tensor = torch.tensor(label, dtype=torch.float32)

        label_sizes_tensor = torch.tensor([self.xmax, self.ymax], dtype=torch.float32)

        sample = {"image": image_tensor, "location": location_tensor,
                  "label": label_tensor, "img_name": img_name.split('/')[-1], "sizes": label_sizes_tensor}

        return sample


def main():
    # dataloader
    data_dir = sys.argv[1]
    testset = images_with_locations(data_dir, 491, 134, 1)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # loader test

    for i_batch, sample_batched in enumerate(testloader):
        print(i_batch, sample_batched['img_name'][0], sample_batched['location'][0])



if __name__ == "__main__":
    main()
