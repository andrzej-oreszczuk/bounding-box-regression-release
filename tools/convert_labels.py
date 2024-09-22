import os
import sys
import cv2
from tools import coordinates
from tools import coordinates_point


def save_img_with_bbox(path_name_img, path_name_label, path_name_label_converted):
    fl = open(path_name_label, "r", encoding="utf-8")
    bbox = fl.readlines()
    fl.close()

    fl = open(path_name_label_converted, "a")

    for dt in bbox:
        points_x = []
        points_y = []

        line_split = dt.split(' ')
        if len(line_split) > 5:
            for d in range(len(line_split)):
                if d > 0:
                    if d % 2 == 1:
                        points_x.append(float(line_split[d]))
                    else:
                        points_y.append(float(line_split[d]))

            min_x = min(points_x)
            max_x = max(points_x)
            min_y = min(points_y)
            max_y = max(points_y)

            # rectangle coordinates:

            x_rect = (min_x + max_x) / 2
            y_rect = (min_y + max_y) / 2
            w_rect = max_x - min_x
            h_rect = max_y - min_y


            label_rect = "0 " + str(x_rect) + " " + str(y_rect) + " " + str(w_rect) + " " + str(h_rect) + "\n"
            fl.write(label_rect)
        else:
            fl.write(dt)

    fl.close()

    return len(bbox)


def main():
    if (len(sys.argv) == 2):
        path_to_imgs = sys.argv[1] + "/images"
        path_to_labels = sys.argv[1] + "/labels"
        path_to_labels_converted = sys.argv[1] + "/labels_converted"
    else:
        path_to_imgs = sys.argv[1]
        path_to_labels = sys.argv[2]
        path_to_labels_converted = sys.argv[3]

    if not os.path.exists(path_to_labels_converted):
        os.mkdir(path_to_labels_converted)

    dir_list = os.listdir(path_to_imgs)

    label_cnt = 0

    for f in dir_list:
        name = f.split(".jpg")[0]
        label_cnt += save_img_with_bbox(path_to_imgs + "/" + f, path_to_labels + "/" + name + ".txt",
                                        path_to_labels_converted + "/" + name + ".txt")

    print("Processed:", len(dir_list), "files and", label_cnt, "labels")


if __name__ == "__main__":
    main()
