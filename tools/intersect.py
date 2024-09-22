import glob
import os
import shutil
import sys

def usage(what = ""):
    print("Usage: python intersect.py setA setB target_directory")
    print("Calculates the intersection of setA and setB in terms of files included, in a non-recursive manner, where setA and setB are paths. Only the 22 leading characters of the filename are compared, but the extensions do not get comapared.")
    print("The files from setA that are in the intersection (and have .jpg extension, only!) get copied to the target_directory.")
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

    letters_to_compare_cnt = 22

    par_cnt = len(sys.argv)
    if par_cnt != 4:
        print("Wrong number of parameters provided")
        return


    pathA = sys.argv[1]
    pathB = sys.argv[2]
    copy_to_path = sys.argv[3]

    if not os.path.exists(pathA) or not os.path.exists(pathB):
        print("Path does not exist")
        return

    if not os.path.exists(copy_to_path):
        os.mkdir(copy_to_path)

    print("setA =", pathA)
    print("setB =", pathB)

    dir_listA_paths = glob.glob(pathA + "/*.*", recursive=False)
    dir_listA_paths.sort(key=filename)
    dir_listB_paths = glob.glob(pathB + "/*.*", recursive=False)
    dir_listB_paths.sort(key=filename)

    dir_listA = [filename(f) for f in dir_listA_paths]
    dir_listB = [filename(f) for f in dir_listB_paths]

    dir_listB.append("ZZZZZZZZZZZZZZZZZZZZZZZZZZZ.jpg")  #plus infinity value

    cnt = indA = indB = 0
    while indA < len(dir_listA):
        # only letters_to_compare_cnt first letters of a filename gets compared
        # because the interesting filename part is of the form SRD_C_09932_0018_01-02

        if indA < len(dir_listA)-1 and dir_listA[indA][0:letters_to_compare_cnt] == dir_listA[indA + 1][0:letters_to_compare_cnt]:
            indA = indA + 1   #skip this word as the same word will be analysed next
        else:
            if dir_listA[indA][0:letters_to_compare_cnt] < dir_listB[indB][0:letters_to_compare_cnt]:
                indA = indA + 1
            elif dir_listA[indA][0:letters_to_compare_cnt] == dir_listB[indB][0:letters_to_compare_cnt]:
                cnt = cnt + 1
                found_filename = dir_listA[indA][0:letters_to_compare_cnt] + ".jpg"
                print(found_filename)
                shutil.copyfile("/".join([directories(dir_listA_paths[indA]), found_filename]),
                                    "/".join([copy_to_path, found_filename]))
                indA = indA + 1
            else: # dir_listA[indA][0:letters_to_compare_cnt] > dir_listB[indB][0:letters_to_compare_cnt]:
                indB = indB + 1   #it is always possible thanks to plus infinity in dir_listB


    print("Found", cnt, "files in intersection of setA and setB.")


if __name__ == "__main__":
    main()
