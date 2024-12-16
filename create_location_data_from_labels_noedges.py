import os
import sys
import random
import cv2
import numpy as np
from tools import coordinates, recoordinate
from tools import coordinates_point



def create_location_data(path_name_label, path_name_location_data, name_img):
    fl = open(path_name_label, "r", encoding="utf-8")
    bbox = fl.readlines()
    fl.close()

    i = 0
    for dt in bbox:
        i += 1
        _, x, y, w, h = map(float, dt.split(' '))
        if (x-w/2) > 0.01 and (x+w/2) < 0.99 and (y-h/2) > 0.01 and (y+h/2) < 0.99:
            fl = open(path_name_location_data + "_" + str(i) + ".txt", "a")

            #generated label
            xl = x
            yl = y
            wl = w
            hl = h

            #generate top of the label
            dt = 1
            s = 1
            while(y - h / 2 - dt < 0.001):
                s *= 0.95
                dt = (0.1 + random.random() / 2) * s * h
            yl -= dt / 2
            hl += dt
            #generate bottom of the label:
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

            fl.write(name_img + " " + str(i) + "\n")

            generated_location = str(_) + " " + str(xl)+ " " + str(yl) + " " + str(wl) + " " + str(hl) + "\n"
            fl.write(generated_location)
            fl.close()

    return len(bbox)


def main():
    if len(sys.argv) == 2:
        path_to_imgs = sys.argv[1] + "/images"
        path_to_labels = sys.argv[1] + "/labels"
        path_to_location_data = sys.argv[1] + "/approximate_labels"
    else:
        path_to_imgs = sys.argv[1]
        path_to_labels = sys.argv[2]
        path_to_location_data = sys.argv[3]

    if not os.path.exists(path_to_location_data):
        os.mkdir(path_to_location_data)

    dir_list = os.listdir(path_to_imgs)

    random.seed()

    label_cnt = 0

    for f in dir_list:
        name = f.split(".jpg")[0]
        print(f)
        label_cnt += create_location_data(path_to_labels + "/" + name + ".txt",
                                          path_to_location_data + "/" + name, f)

    print("Processed:", len(dir_list), "files and", label_cnt, "labels")


if __name__ == "__main__":
    main()
