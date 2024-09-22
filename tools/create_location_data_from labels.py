import os
import sys
import random
import cv2
from tools import coordinates
from tools import coordinates_point


def create_location_data(path_name_label, path_name_location_data):
    fl = open(path_name_label, "r", encoding="utf-8")
    bbox = fl.readlines()
    fl.close()

    fl = open(path_name_location_data, "a")

    for dt in bbox:
        _, x, y, w, h = map(float, dt.split(' '))
        # simulate approximate manually picked location:
        y_location = y + 0.01 * random.normal(0, 1)
        x_location = x + 0.01 * random.normal(0, 1)

        generated_location = str(x_location) + " " + str(y_location) + "\n"
        fl.write(generated_location)

    fl.close()

    return len(bbox)


def main():
    if len(sys.argv) == 2:
        path_to_imgs = sys.argv[1] + "/images"
        path_to_labels = sys.argv[1] + "/labels"
        path_to_location_data = sys.argv[1] + "/locations_of_objects"
    else:
        path_to_imgs = sys.argv[1]
        path_to_labels = sys.argv[2]
        path_to_location_data = sys.argv[3]

    if not os.path.exists(path_to_location_data):
        os.mkdir(path_to_location_data)

    dir_list = os.listdir(path_to_imgs)

    label_cnt = 0

    for f in dir_list:
        name = f.split(".jpg")[0]
        label_cnt += create_location_data(path_to_labels + "/" + name + ".txt",
                                          path_to_location_data + "/" + name + "locations.txt")

    print("Processed:", len(dir_list), "files and", label_cnt, "labels")


if __name__ == "__main__":
    main()
