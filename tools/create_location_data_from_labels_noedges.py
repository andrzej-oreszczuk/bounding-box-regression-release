import os
import sys
import random
import cv2
import numpy as np
from tools import coordinates
from tools import coordinates_point



def create_location_data(path_name_label, path_name_location_data, name_img):
    fl = open(path_name_label, "r", encoding="utf-8")
    bbox = fl.readlines()
    fl.close()

    i = 0
    for dt in bbox:
        i += 1
        _, x, y, w, h = map(float, dt.split(' '))
        if (x-w/2) > 0.00001 and (x+w/2) < 0.9999 and (y-h/2) > 0.00001 and (y+h/2) < 0.9999:
            fl = open(path_name_location_data + "_" + str(i) + ".txt", "a")

            # simulate approximate manually picked location:
            y_location = y + 0.002 * np.random.normal(0, 1)
            x_location = x + 0.002 * np.random.normal(0, 1)

            fl.write(name_img + " " + str(i) + "\n")

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
        print(f)
        label_cnt += create_location_data(path_to_labels + "/" + name + ".txt",
                                          path_to_location_data + "/" + name, f)

    print("Processed:", len(dir_list), "files and", label_cnt, "labels")


if __name__ == "__main__":
    main()
