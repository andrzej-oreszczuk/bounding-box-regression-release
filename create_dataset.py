import os
import sys
import cv2
import random
import math
from tools import coordinates

def random_color():
    if random.random() > 0.5:
        r = random.randrange(30, 80)
    else:
        r = random.randrange(0, 50)
    return ((r + random.randrange(20), r + random.randrange(20), r + random.randrange(20)),
            (r + random.randrange(20) + 20, r + random.randrange(20) + 20, r + random.randrange(20) + 20))


def brighten(img, val):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,val)
    v[v > 255] = 255 - random.randrange(20)
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def draw_ellipse(image, x, y, a, b, w, f, thickness):



    color = random_color()
    if f == 1:
        cv2.ellipse(image, (x, y), (a, b), w, 0, 360, color[1], -1)

    cv2.ellipse(image, (x, y), (round(a-thickness/2)-1, round(b-thickness/2)-1), w, 0, 360, color[0], thickness)

def draw_quadrangle(image, l, t, r, b, w, h, f, thickness):
    color = random_color()
    tls = (round(l + random.random() * w), round(t + thickness/2))
    trs = (round(r - thickness/2), round(t + random.random() * h))
    brs = (round(r - random.random() * w), round(b - thickness/2))
    bls = (round(l + thickness/2), round(t + random.random() * h))

    xy = (round((r+l)/2), round((b + t) / 2))

    cv2.line(image, tls, trs, random.choice(color), thickness)
    cv2.line(image, trs, brs, random.choice(color), thickness)
    cv2.line(image, brs, bls, random.choice(color), thickness)
    cv2.line(image, bls, tls, random.choice(color), thickness)
    if f == 1:
        cv2.line(image, xy, trs, random.choice(color), thickness)
        cv2.line(image, xy, brs, random.choice(color), thickness)
        cv2.line(image, xy, bls, random.choice(color), thickness)
        cv2.line(image, xy, tls, random.choice(color), thickness)

def draw_hash(image, l, t, r, b, w, h, nmax, thickness):
    color = random_color()

    l = round(l + thickness/2)
    r = round(r - thickness / 2)
    t = round(t + thickness / 2)
    b = round(b - thickness / 2)

    nh = random.randrange(2, nmax)
    nw = random.randrange(2, nmax)

    dw = round(w/(nw+1))
    dh = round(h/(nh+1))



    for i in range(nh):
        cv2.line(image, (l, t + i * dh + random.randrange(dh)), (r, t + i * dh + random.randrange(dh)), random.choice(color), thickness)

    for i in range(nw):
        cv2.line(image, (l + i * dw + random.randrange(dw), b), (l + i * dw + random.randrange(dw), t), random.choice(color), thickness)



def save_img_with_bbox(path_name_img, path_name_label, path_name_img_with_frame):
    
    img = cv2.imread(path_name_img)
    dh, dw, _ = img.shape
    fl = open(path_name_label, "r", encoding = "utf-8")
    bbox = fl.readlines()
    fl.close()

    img = brighten(img, 150)

    for dt in bbox:

        _, x, y, w, h = map(float, dt.split(' '))

        l, t, r, b = coordinates(x, y, w, h, dw, dh)
        #cv2.rectangle(img, (l, t), (r, b), (255, 0, 0), 1)

        x = round(x * dw)
        w = round(w * dw)
        y = round(y * dh)
        h = round(h * dh)

        s = random.random()

        if s < 0.33:
            draw_quadrangle(img, l, t, r, b, w, h, random.randrange(1, 3), random.randrange(3, 10))
        elif s > 0.67:
            a = 0
            draw_ellipse(img, x, y, round(w/2), round(h/2), a, random.randrange(1, 3), random.randrange(3, 10))
        else:
            draw_hash(img, l, t, r, b, w, h, 7, random.randrange(3, 10))

    cv2.imwrite(path_name_img_with_frame, img)
    
    return len(bbox)

def main():
    if (len(sys.argv) == 2):
        path_to_imgs = sys.argv[1]+"/images"
        path_to_labels = sys.argv[1]+"/labels"
        path_to_imgs_with_frame = sys.argv[1]+"/images_with_frames"
    else:
        path_to_imgs = sys.argv[1]
        path_to_labels = sys.argv[2]
        path_to_imgs_with_frame = sys.argv[3]
    
    if not os.path.exists(path_to_imgs_with_frame):
        os.mkdir(path_to_imgs_with_frame)

    random.seed()
    dir_list = os.listdir(path_to_imgs)
    
    label_cnt = 0
    for f in dir_list:
        name = f.split(".jpg")[0]
        print(name)
        label_cnt += save_img_with_bbox(path_to_imgs+"/"+f, path_to_labels+"/"+name+".txt", path_to_imgs_with_frame+"/"+f)
    
    print("Processed:",  len(dir_list), "files and", label_cnt, "labels")
    
if __name__ == "__main__":
    main()
