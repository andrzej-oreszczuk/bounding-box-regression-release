import glob
import sys
import os
import cv2

def split(filename, margin, write_path):
#image_slicer.slice(x, 4)
    img = cv2.imread(filename)
    Y = int((img.shape[0]//2) * (1 + margin))
    X = img.shape[1]//2

    tiles = [img[y:y+Y,x:x+X] for y in [0, img.shape[0] - Y] for x in [0, X]]

      #saving
    name_jpg = filename.replace("\\", "/").split("/")[-1]
    name = name_jpg.split(".jpg")[0]
    for i in range(4):
      cv2.imwrite(write_path+"/%s_%s.jpg"%(name, ["01-01", "02-01", "01-02", "02-02"][i]), tiles[i])


def split_path(read_path, margin, write_path):
  filelist = glob.glob(read_path+"/*.jpg")
  for filename in filelist:
    split(filename, margin, write_path)
  return len(filelist)

def main():

  margin = 0.03

  if (len(sys.argv) == 3):
    read_path = sys.argv[1]
    write_path = sys.argv[2]

  if not os.path.exists(write_path):
    os.mkdir(write_path)

  cnt = 1
  if os.path.isdir(read_path):
    cnt = split_path(read_path, margin, write_path)
  else:
    split(read_path, margin, write_path)

  print("Processed:", cnt, "files")


if __name__ == "__main__":
  main()