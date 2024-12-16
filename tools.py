
def coordinates(x, y, w, h, img_width, img_heigth, shift=0):
    l = round((x - w / 2) * (img_width)) - shift
    r = round((x + w / 2) * (img_width)) + shift
    t = round((y - h / 2) * (img_heigth)) - shift
    b = round((y + h / 2) * (img_heigth)) + shift
    
    if l < 0:
        l = 0
    if t < 0:
        t = 0
    if r > img_width-1:
        r = img_width-1
    if b > img_heigth-1:
        b = img_heigth-1

    return (l, t, r, b)


def coordinates_point(xs, ys, img_width, img_heigth, shift=0):
    x = round((xs) * (img_width)) - shift
    y = round((ys) * (img_width)) + shift

    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x > img_width - 1:
        x = img_width - 1
    if y > img_heigth - 1:
        y = img_heigth - 1

    return (x, y)

def recoordinate(l, t, r, b, img_width, img_heigth):
    x = (l + r) / 2 / (img_width)
    y = (b + t) / 2 / (img_heigth)
    w = (r - l) / (img_width)
    h = (b - t) / (img_heigth)

    return (x, y, w, h)
