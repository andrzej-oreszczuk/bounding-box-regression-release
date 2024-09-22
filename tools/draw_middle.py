import os
import sys
import cv2
from tools import coordinates




def save_img_with_bbox(path_name_img, path_name_img_with_frame):
    if os.path.exists(path_name_img):

        img = cv2.imread(path_name_img)
        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        dh, dw, _ = img.shape

        print((dw/2-3, dh/2-3))

        cv2.rectangle(img, (int(dw/2)-3, int(dh/2)-3), (int(dw/2)+3, int(dh/2)+3), (0, 0, 255), 2)

        cv2.imwrite(path_name_img_with_frame, img)

        return 1
    else:
        return 0

def main():
    path_to_images = sys.argv[1]
    path_to_images_frames = sys.argv[2]
    dir_list = os.listdir(path_to_images)

    if not os.path.exists(path_to_images_frames):
        os.mkdir(path_to_images_frames)

    for f in dir_list:
        name = f
        save_img_with_bbox(path_to_images+"/"+name, path_to_images_frames+"/"+name)

    
if __name__ == "__main__":
    main()
