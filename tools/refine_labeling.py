# img_viewer.py
from tkinter import Image

from PIL import Image, ImageDraw
import PySimpleGUI as sg
import os.path
import cv2
import io
from tools import coordinates, recoordinate

# First the window layout in 2 columns

############### GLOBAL VARIABLES ############################

labels = []
cur_label = 0
fnames = []
width, height = 0, 0

def move_bb(dl, dt, dr, db):
    global width
    global height
    global cur_label
    l, t, r, b = coordinates(*(labels[cur_label]), width, height)
    labels[cur_label] = recoordinate(l+dl, t+dt, r+dr, b+db, width, height)
    save_labels()
    redraw_label_on_image()
def label_filename():
    return os.path.join(values["-FOLDER-"], "labels", listbox_filename().replace("\\", "/").split("/")[-1].split(".jpg")[0] + ".txt")

def save_labels():
    with open(label_filename(), "w", encoding="ANSI") as label_file:
        for label in labels:
            label_file.write("0 " + " ".join([str(i) for i in label])+"\n")


def update_label_text():
    window["-LA_NUM-"].update("label " + str(cur_label + 1) + " of " + str(len(labels)))

def listbox_selection():
    lb = window["-FILE LIST-"].widget
    x1 = lb.curselection()[0]
    return x1, lb

def listbox_filename():
    x1, _ = listbox_selection()

    filename = os.path.join(
        values["-FOLDER-"], "images", fnames[x1]  # values["-FILE LIST-"][0]
    )
    return filename
def redraw_label_on_image():
    global labels
    global cur_label
    global width
    global height

    filename = listbox_filename()
    img = Image.open(filename)
    width, height = img.size
    l, t, r, b = coordinates(*(labels[cur_label]), width, height)
    draw = ImageDraw.Draw(img)  # modifies my img inplace
    draw.rectangle([l, t, r, b], outline="red")

    l, t, r, b = coordinates(*(labels[cur_label]), width, height, shift=50)
    img = img.crop((l, t, r, b))

    bio = io.BytesIO()  # a binary memory resident stream
    img.save(bio, format='PNG')  # save image as png to it
    imgbytes = bio.getvalue()  # this can be used by OpenCV hopefully
    window["-IMAGE-"].update(data=imgbytes)

def move_down():   # https://stackoverflow.com/questions/50148237/how-to-move-item-of-listbox-by-python-tkinter
    x1, lb = listbox_selection()
    if x1 + 1 == lb.size():
        pass #lb.selection_set(0)
    else:
        lb.selection_clear(x1)
        lb.selection_set(x1 + 1)
        #window["-FILE LIST-"].Update(set_to_index=x1+1)   #https://github.com/PySimpleGUI/PySimpleGUI/issues/1278

def move_up():
    x1, lb = listbox_selection()
    if x1 - 1 < 0:
        pass #lb.selection_set(0)
    else:
        lb.selection_clear(x1)
        lb.selection_set(x1 - 1)

def new_file_in_filelist(last_label=False):
    global labels
    global cur_label
    ############## FILE NAME ##############################

    filename = listbox_filename()

    filename_fragments = filename.replace("\\", "/").split("/")
    window["-IM_NAME-"].update(filename_fragments[-1])

    ############## LABELS ##############################

    labels = []
    with open(label_filename(), "r", encoding="ANSI") as label_file:
        lines = label_file.readlines()
        for line in lines:
            _, x, y, w, h = map(float, line.split(' '))
            labels.append([x, y, w, h])

    if last_label:
        cur_label = len(labels)-1
    else:
        cur_label = 0
    update_label_text()

    ############## IMAGE ##############################
    redraw_label_on_image()

file_list_column = [
    [
        sg.Text("Images/Labels Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Text("/images")
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
    [
        sg.Button("Previous label", key="-PLABEL-"), sg.Button("Next label", key="-NLABEL-")
    ],
    [
        sg.Text(size=(20, 1))
    ],
    [
        sg.Text(size=(20, 1))
    ],
    [
        sg.Text(size=(10, 1)), sg.Button("Top Line ^", key="-T-LINE-U-"), sg.Button("Top Line v", key="-T-LINE-D-")
    ],
    [
        sg.Button("Left Line <", key="-L-LINE-L-"), sg.Button("Left Line >", key="-L-LINE-R-"),
        sg.Button("Right Line <", key="-R-LINE-L-"), sg.Button("Right Line >", key="-R-LINE-R-"),
    ],
    [
        sg.Text(size=(10, 1)), sg.Button("Bottom Line ^", key="-B-LINE-U-"), sg.Button("Bottom Line v", key="-B-LINE-D-")
    ],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(40, 1), key="-IM_NAME-")],
    [sg.Text(size=(80, 1), key="-LA_NUM-")],
    [sg.Image(key="-IMAGE-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("Image Viewer", layout)




# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            new_file_in_filelist()
        except ():
            pass
    elif event == "-NLABEL-":
        if cur_label == len(labels) - 1:
            move_down()
            new_file_in_filelist()
        else:
            cur_label = cur_label + 1
            update_label_text()
            redraw_label_on_image()
    elif event == "-PLABEL-":
        if cur_label == 0:
            move_up()
            new_file_in_filelist(last_label=True)
        else:
            cur_label = cur_label - 1
            update_label_text()
            redraw_label_on_image()
    elif event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder+"/images")
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder+"/images", f))
            and f.lower().endswith(".jpg")
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-T-LINE-U-":
        move_bb(0, -1, 0, 0)
    elif event == "-T-LINE-D-":
        move_bb(0, +1, 0, 0)
    elif event == "-B-LINE-U-":
        move_bb(0, 0, 0, -1)
    elif event == "-B-LINE-D-":
        move_bb(0, 0, 0, +1)
    elif event == "-L-LINE-L-":
        move_bb(-1, 0, 0, 0)
    elif event == "-L-LINE-R-":
        move_bb(+1, 0, 0, 0)
    elif event == "-R-LINE-L-":
        move_bb(0, 0, -1, 0)
    elif event == "-R-LINE-R-":
        move_bb(0, 0, +1, 0)

window.close()