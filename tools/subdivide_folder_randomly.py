import glob
import os
import shutil
import sys
import random

def usage(subfolders_cnt):
    print("Usage: python subdivide_folder_randomly.py source_directory target_directory")
    print("The files from the source_directory get copied to the target_directory subdirectory 1-%d, randomly."%subfolders_cnt)
    print("")
    return

def filename(path):
    return path.replace("\\", "/").split("/")[-1]

def directories(path):
    components = path.replace("\\", "/").split("/")
    return "/".join(components[0:len(components)-1])

def main():
    subfolders_cnt = 5

    usage(subfolders_cnt)

    par_cnt = len(sys.argv)
    if par_cnt != 3:
        print("Wrong number of parameters provided")
        return


    source_path = sys.argv[1]
    target_path = sys.argv[2]


    if not os.path.exists(source_path):
        print("Source path does not exist")
        return

    if not os.path.exists(target_path):
        os.mkdir(target_path)

    for i in range(1, subfolders_cnt+1):
        if not os.path.exists(target_path+"/%d"%i):
            os.mkdir(target_path+"/%d"%i)

    source_files = glob.glob(source_path + "/*.*", recursive=False)

    random.seed(0)
    copied = 0
    for _, file_path in enumerate(source_files):
        copied = copied + 1
        shutil.copyfile(file_path, "/".join([target_path+"/%d"%random.randint(1, subfolders_cnt), filename(file_path)]) )


    print("Copied", copied, "files from %s into %s"%(source_path, target_path))

if __name__ == "__main__":
    main()
