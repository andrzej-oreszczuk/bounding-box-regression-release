import glob
import os
import shutil
import sys

def usage(what = ""):
    print("Usage: python subtract.py setA setB [<check_labels> or target_directory]")
    print("Calculates setA-setB in terms of .jpg files, where setA and setB are paths, recursively. Only the 16 leading characters of the filename are compared.")
    print("If `check_labels` string is provided as the 3rd parameter, it calculates the difference in terms of labels too, and setA and setB paths should point to a directory with /labels and /images subdirectories")
    print("Alternatively, if target directory is provided as the 3rd parameter, the .jpg files (16-character name only) consituting the difference will be copied into this directory.")
    print("")
    print(what)
    return

def filename(path):
    return path.replace("\\", "/").split("/")[-1]

def directories(path):
    components = path.replace("\\", "/").split("/")
    return "/".join(components[0:len(components)-1])

def main():
    usage()

    letters_to_compare_cnt = 16

    par_cnt = len(sys.argv)
    if par_cnt != 3 and par_cnt != 4:
        print("Wrong number of parameters provided")
        return

    check_labels = False
    should_copy = False
    if par_cnt == 4:
        if sys.argv[3] == "check_labels":
            check_labels = True
        else:
            should_copy = True
            copy_to_path = sys.argv[3]
            if not os.path.exists(copy_to_path):
                os.mkdir(copy_to_path)

    pathA_img = pathA = sys.argv[1]
    pathB_img = pathB = sys.argv[2]

    if check_labels:
        print("Checking labels!")
        pathA_img = pathA + "/images"
        pathB_img = pathB + "/images"


    if not os.path.exists(pathA_img) or not os.path.exists(pathB_img):
        print("Path does not exist")
        return

    print("setA =", pathA_img)
    print("setB =", pathB_img)

    dir_listA_paths = glob.glob(pathA_img + "/**/*.jpg", recursive=True)
    dir_listA_paths.sort(key=filename)
    dir_listB_paths = glob.glob(pathB_img + "/**/*.jpg", recursive=True)
    dir_listB_paths.sort(key=filename)

    dir_listA = [filename(f) for f in dir_listA_paths]
    dir_listB = [filename(f) for f in dir_listB_paths]

    dir_listB.append("ZZZZZZZZZZZZZZZZZZZZZZZZZZZ.jpg")  #plus infinity value

    cnt = indA = indB = 0
    while indA < len(dir_listA):
        # only letters_to_compare_cnt first letters of a filename gets compared
        # because the interesting filename part is of the form SRD_C_09932_0018

        if indA < len(dir_listA)-1 and dir_listA[indA][0:letters_to_compare_cnt] == dir_listA[indA + 1][0:letters_to_compare_cnt]:
            indA = indA + 1   #skip this word as the same word will be analysed next
        else:
            if dir_listA[indA][0:letters_to_compare_cnt] < dir_listB[indB][0:letters_to_compare_cnt]:
                cnt = cnt + 1
                found_filename = dir_listA[indA][0:letters_to_compare_cnt]+".jpg"
                print(found_filename)
                if should_copy:
                    shutil.copyfile("/".join([directories(dir_listA_paths[indA]),found_filename]), "/".join([copy_to_path, found_filename]))
                indA = indA + 1
            elif dir_listA[indA][0:letters_to_compare_cnt] == dir_listB[indB][0:letters_to_compare_cnt]:
                if check_labels:
                    with open(pathA+"/labels/"+dir_listA[indA].split(".jpg")[0]+".txt", "r", encoding="ANSI") as f:
                        linesA = f.readlines()
                    with open(pathB + "/labels/" + dir_listA[indA].split(".jpg")[0]+".txt", "r", encoding="ANSI") as f:
                        linesB = f.readlines()
                    if len(linesA) > len(linesB):
                        print(dir_listA[indA], "(", len(linesA), "labels in A;", len(linesB), "labels in B )")
                        cnt = cnt + 1
                indA = indA + 1
            else: # dir_listA[indA][0:letters_to_compare_cnt] > dir_listB[indB][0:letters_to_compare_cnt]:
                indB = indB + 1   #it is always possible thanks to plus infinity in dir_listB


    print("Found", cnt, "files present in setA and not present in setB"+(", including labels check" if check_labels else ""))


if __name__ == "__main__":
    main()
