import os
import sys
import random
import numpy as np
from PIL import Image
from tools import coordinates
from tools import coordinates_point



def analysis(path_name_label, xi, yi):
    fl = open(path_name_label, "r", encoding="utf-8")
    bbox = fl.readlines()
    fl.close()

    xmax = 0
    ymax = 0
    for dt in bbox:

        _, x, y, w, h = map(float, dt.split(' '))
        if (x - w / 2) > 0.00001 and (x + w / 2) < 0.9999 and (y - h / 2) > 0.00001 and (y + h / 2) < 0.9999:

            xw = round(xi*w)
            yh = round(yi*h)

            if xw > xmax:
                xmax = xw
            if yh > ymax:
                ymax = yh

    return xmax, ymax

def main():

    dataset_root = sys.argv[1]

    xmax = 0
    ymax = 0

    set = dataset_root + "/train"
    dir_list = os.listdir(set + "/images")
    for f in dir_list:
        image = Image.open(set + "/images/" + f).convert('L')
        xi, yi = image.size
        image.close()
        name = f.split(".jpg")[0]
        x, y = analysis(set + "/labels" + "/" + name + ".txt", xi, yi)
        if x > xmax:
            xmax = x
        if y > ymax:
            ymax = y

    set = dataset_root + "/valid"
    dir_list = os.listdir(set + "/images")
    for f in dir_list:
        image = Image.open(set + "/images/" + f).convert('L')
        xi, yi = image.size
        image.close()
        name = f.split(".jpg")[0]
        x, y = analysis(set + "/labels" + "/" + name + ".txt", xi, yi)
        if x > xmax:
            xmax = x
        if y > ymax:
            ymax = y

    set = dataset_root + "/test"
    dir_list = os.listdir(set + "/images")
    for f in dir_list:
        image = Image.open(set + "/images/" + f).convert('L')
        xi, yi = image.size
        image.close()
        name = f.split(".jpg")[0]
        x, y = analysis(set + "/labels" + "/" + name + ".txt", xi, yi)
        if x > xmax:
            xmax = x
        if y > ymax:
            ymax = y

    print(xmax, ymax)

if __name__ == "__main__":
    main()
