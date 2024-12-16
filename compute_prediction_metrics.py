import os
import sys
import cv2
from tools import coordinates


def compute_object_metrics(root_dir, path_to_imgs, path_to_predictions, f):
    fl = open(f, "r", encoding="utf-8")
    lines = fl.readlines()
    fl.close()

    first_line = lines[0].split()

    label_number = first_line[-1]
    image_name = lines[0].rstrip()[:-(len(label_number) + 1)]
    path_name_img = path_to_imgs + "/" + image_name

    labels_file = root_dir + "/labels/" + image_name[:-4] + ".txt"
    predictions_file = path_to_predictions + "/" + image_name[:-4] + ".txt"

    fl = open(labels_file)
    labels_data = fl.readlines()
    fl.close()

    if not os.path.exists(predictions_file):
        return 0

    fl = open(predictions_file)
    predictions_data = fl.readlines()
    fl.close()

    label_str = labels_data[int(label_number) - 1]

    # ground truth label
    dt = label_str

    _, x, y, w, h = map(float, dt.split(' '))

    # find closest predicted label
    xc, yx, wc, hc = 0, 0, 0, 0

    mindistance = 2
    for prediction in predictions_data:
        dtp = prediction

        _p, xp, yp, wp, hp = map(float, dtp.split(' '))

        distance = (x - xp) ** 2 + (y - yp) ** 2
        if distance < mindistance:
            mindistance = distance
            xc = xp
            yc = yp
            wc = wp
            hc = hp

    img = cv2.imread(path_name_img)
    dh, dw, _ = img.shape

    #prediction coordinates
    lp, tp, rp, bp = coordinates(xc, yc, wc, hc, dw, dh)

    # label coordinates
    l, t, r, b = coordinates(x, y, w, h, dw, dh)

    # check if middle of prediction is inside true label:
    if (lp+rp)/2 > r or (lp+rp)/2 < l or (tp+bp)/2 > b or (tp+bp)/2 < t:
        return 0

    pe_width = abs(r - l - rp + lp) / (r - l)
    absolute_error_pixel_width = abs(l - lp + rp - r)
    pe_height = abs(b - t - bp + tp) / (b - t)
    absolute_error_pixel_height = abs(t - tp + bp - b)

    mae_pixel = (abs(rp - r) + abs(lp - l) + abs(bp - b) + abs(tp - t))/4


    return pe_width, absolute_error_pixel_width, pe_height, absolute_error_pixel_height, mae_pixel


def main():

    path_to_predictions = sys.argv[1]
    root_dir = sys.argv[2]
    path_to_imgs = root_dir + "/images"

    dir_list = os.listdir(root_dir + "/approximate_labels")
    label_cnt = 0

    mape_width = 0
    mae_pixel_width = 0
    mape_height = 0
    mae_pixel_height = 0
    mae = 0
    n = 0

    for f in dir_list:
        metrics = compute_object_metrics(root_dir, path_to_imgs, path_to_predictions,
                                        root_dir + "/approximate_labels/" + f)
        print(f)
        if metrics == 0:
            print("label ", f, " not in predictions")
            label_cnt+=1
        else:
            n += 1
            mape_width += metrics[0]
            mae_pixel_width += metrics[1]
            mape_height += metrics[2]
            mae_pixel_height += metrics[3]
            mae += metrics[4]
    print("mape szerokości (mae szerokości w pikselach): ", mape_width/n, "(", mae_pixel_width/n, ")\n", "mape wysokości (mae wysokości w pikselach): ",
          mape_height/n, "(", mae_pixel_height/n, ")\n", "mae w pikselach: ",
          mae/n, "\n", label_cnt, " obiektów bez przewidzianej ramki")



if __name__ == "__main__":
    main()
