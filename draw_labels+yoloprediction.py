import os
import sys
import cv2
from tools import coordinates




def save_img_with_bbox(root_dir, path_to_imgs, path_to_yolo_predictions, f):

    fl = open(f, "r", encoding="utf-8")
    lines = fl.readlines()
    fl.close()

    first_line = lines[0].split()

    label_number = first_line[-1]
    image_name = lines[0].rstrip()[:-(len(label_number) + 1)]
    path_name_img = path_to_imgs + "/" + image_name

    labels_file = root_dir + "/labels/" + image_name[:-4] + ".txt"
    predictions_file = path_to_yolo_predictions + "/" + image_name[:-4] + ".txt"


    fl = open(labels_file)
    labels_data = fl.readlines()
    fl.close()

    if not os.path.exists(predictions_file):
        return 0

    fl = open(predictions_file)
    predictions_data = fl.readlines()
    fl.close()

    label_str = labels_data[int(label_number) - 1]

    #ground truth label
    dt = label_str

    _, x, y, w, h = map(float, dt.split(' '))

    #find closest predicted label
    xc, yx, wc, hc = 0, 0, 0, 0

    mindistance = 2
    for prediction in predictions_data:
        dtp = prediction

        _p, xp, yp, wp, hp = map(float, dtp.split(' '))

        distance = (x-xp)**2 + (y-yp)**2
        if distance < mindistance:
            mindistance = distance
            xc = xp
            yc = yp
            wc = wp
            hc = hp


    img = cv2.imread(path_name_img)
    dh, dw, _ = img.shape



    #according to this image: https://github.com/center4ml/Object_detection_tutorial/blob/main/01_Datasets/_resourses/image-25.png
    #0 maps as 0, 1 maps as (dw-1)
    l, t, r, b = coordinates(xc, yc, wc, hc, dw, dh)

    cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 1)

    cv2.imwrite(path_to_imgs + "/" + image_name, img)

    return 1


def main():

    path_to_imgs = sys.argv[1]
    path_to_yolo_predictions = sys.argv[2]
    root_dir = sys.argv[3]

    
    dir_list = os.listdir(root_dir+"/approximate_labels")
    label_cnt = 0
    for f in dir_list:
        label_cnt += save_img_with_bbox(root_dir, path_to_imgs, path_to_yolo_predictions,root_dir+"/approximate_labels/"+f)
        print(label_cnt)
    
    print("Processed:",  len(dir_list), "files and", label_cnt, "labels")
    
if __name__ == "__main__":
    main()
