# bounding-box-regression-release

This repo contains a neural net that performs bounding box regression on imprecisely labeled bounding boxes, and associated tools. After training on precisely labeled data it can perform bounding box regression on provided imprecise labels, and at least for some datasets (microscopy images of chloroplast structures in leaves) it performs better than the latest YOLOv11 model with a comparable number of parameters.


Scripts bounding_box_train.py, bounding_box_create_labels.py, bounding_box_detect_angle.py are used to train the model, predict bounding boxes, predict the angle of rotation for approximately rectangular objects. 
Each of them needs a configuration file (eg. python3 bounding_box_train.py config_train.py). Configuration arguments are described in the configuration files.

Model architecture is defined in init_model.py, training parameters in lightning_module.py.

Tools catalouge contains various tools, including create_location_data_from_labels.py, which creates in a given catalogue with a YOLO format dataset with images and labels catalogues an additional catalogue with objects locations used for model training.
