import os
import sys
import cv2

import random
from tools import coordinates


def modify_labels(path_name_img, path_name_label):
    img = cv2.imread(path_name_img)
    dh, dw, _ = img.shape
    fl = open(path_name_label, "r", encoding="utf-8")
    bbox = fl.readlines()
    fl.close()

    fl = open(path_name_label, "w", encoding="utf-8")


    for dt in bbox:
        _, x, y, w, h = map(float, dt.split(' '))

        w -= w / 2 * random.random()
        h -= h / 2 * random.random()
        line = str(_) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n"
        fl.write(line)

    fl.close()



    return len(bbox)


def main():
    if (len(sys.argv) == 2):
        path_to_imgs = sys.argv[1] + "/images"
        path_to_labels = sys.argv[1] + "/labels"
    else:
        path_to_imgs = sys.argv[1]
        path_to_labels = sys.argv[2]


    random.seed()

    dir_list = os.listdir(path_to_imgs)

    label_cnt = 0
    for f in dir_list:
        name = f.split(".jpg")[0]
        label_cnt += modify_labels(path_to_imgs + "/" + f, path_to_labels + "/" + name + ".txt")

    print("Processed:", len(dir_list), "files and", label_cnt, "labels")


if __name__ == "__main__":
    main()
