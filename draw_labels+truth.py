import os
import sys
import cv2
from tools import coordinates




def save_img_with_bbox(root_dir, path_to_imgs, path_to_imgs_with_frame, f):

    fl = open(f, "r", encoding="utf-8")
    lines = fl.readlines()
    fl.close()

    first_line = lines[0].split()

    label_number = first_line[-1]
    image_name = lines[0].rstrip()[:-(len(label_number) + 1)]
    path_name_img = path_to_imgs + "/" + image_name

    labels_file = root_dir + "/labels/" + image_name[:-4] + ".txt"



    fl = open(labels_file)
    labels_data = fl.readlines()
    fl.close()

    label_str = labels_data[int(label_number) - 1]

    img = cv2.imread(path_name_img)
    dh, dw, _ = img.shape

    dt = label_str

    _, x, y, w, h = map(float, dt.split(' '))

    #according to this image: https://github.com/center4ml/Object_detection_tutorial/blob/main/01_Datasets/_resourses/image-25.png
    #0 maps as 0, 1 maps as (dw-1)
    l, t, r, b = coordinates(x, y, w, h, dw, dh)

    cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 3)

    cv2.imwrite(path_to_imgs + "/" + image_name, img)

    return 1


def main():

    path_to_imgs = sys.argv[1]
    path_to_imgs_with_frame = sys.argv[2]
    root_dir = sys.argv[3]
    
    if not os.path.exists(path_to_imgs_with_frame):
        os.mkdir(path_to_imgs_with_frame)
    
    dir_list = os.listdir(root_dir+"/approximate_labels")
    label_cnt = 0
    for f in dir_list:
        label_cnt += save_img_with_bbox(root_dir, path_to_imgs, path_to_imgs_with_frame,root_dir+"/approximate_labels/"+f)
        print(label_cnt)
    
    print("Processed:",  len(dir_list), "files and", label_cnt, "labels")
    
if __name__ == "__main__":
    main()
