import os
import sys
import cv2
from tools import coordinates, recoordinate




def save_label(root_dir, path_to_created_labels, f):

    fl = open(f, "r", encoding="utf-8")
    lines = fl.readlines()
    fl.close()

    first_line = lines[0].split()

    img_name = first_line[0]
    label_number = first_line[1]

    labels_file = root_dir + "/labels/" + img_name.rstrip()[:-4] + ".txt"

    fl = open(labels_file)
    labels_data = fl.readlines()
    fl.close()

    label_str = labels_data[int(label_number) - 1]

    fl = open(path_to_created_labels + "/" + img_name.rstrip()[:-4] + ".txt", "a")
    line = label_str + "\n"
    fl.write(line)
    fl.close()

    return 1


def main():

    root_dir = sys.argv[1]
    path_to_created_labels = root_dir + "/labels_used"
    
    if not os.path.exists(path_to_created_labels):
        os.mkdir(path_to_created_labels)
    
    dir_list = os.listdir(root_dir+"/locations_of_objects")
    label_cnt = 0
    for f in dir_list:
        label_cnt += save_label(root_dir, path_to_created_labels, root_dir+"/locations_of_objects/"+f)
    
    print("Processed:",  len(dir_list), "files and", label_cnt, "labels")
    
if __name__ == "__main__":
    main()
