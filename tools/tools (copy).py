import os
import sys
import cv2
from tools import coordinates


def save_img_with_bbox(path_name_img, path_name_label, path_name_img_with_frame):
    img = cv2.imread(path_name_img)
    dh, dw, _ = img.shape
    fl = open(path_name_label, "r", encoding="utf-8")
    bbox = fl.readlines()
    fl.close()

    for dt in bbox:
        _, x, y, w, h = map(float, dt.split(' '))

        # according to this image: https://github.com/center4ml/Object_detection_tutorial/blob/main/01_Datasets/_resourses/image-25.png
        # 0 maps as 0, 1 maps as (dw-1)
        l, t, r, b = coordinates(x, y, w, h, dw, dh)

        cv2.rectangle(img, (l, t), (r, b), (255, 0, 0), 1)

    cv2.imwrite(path_name_img_with_frame, img)

    return len(bbox)


def main():
    if (len(sys.argv) == 2):
        path_to_imgs = sys.argv[1] + "/images"
        path_to_labels = sys.argv[1] + "/labels"
        path_to_imgs_with_frame = sys.argv[1] + "/images_with_frames"
    else:
        path_to_imgs = sys.argv[1]
        path_to_labels = sys.argv[2]
        path_to_imgs_with_frame = sys.argv[3]

    if not os.path.exists(path_to_imgs_with_frame):
        os.mkdir(path_to_imgs_with_frame)

    dir_list = os.listdir(path_to_imgs)

    label_cnt = 0
    for f in dir_list:
        name = f.split(".jpg")[0]
        label_cnt += save_img_with_bbox(path_to_imgs + "/" + f, path_to_labels + "/" + name + ".txt",
                                        path_to_imgs_with_frame + "/" + f)

    print("Processed:", len(dir_list), "files and", label_cnt, "labels")


if __name__ == "__main__":
    main()
