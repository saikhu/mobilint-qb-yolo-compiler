# 01_prepare_calib_dataset.py
# Runs once one until the dataset does not change

import cv2
import numpy as np 
from qubee.calibration import make_calib_man

img_size = [1280, 1280]

def preprocess_yolo(image_path: str):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h0, w0 = img.shape[:2]  # Original hw
    r = min(img_size[0] / h0, img_size[1] / w0)  # ratio
    new_unpad = int(round(w0 * r)), int(round(h0 * r))
    dh, dw = (img_size[0] - new_unpad[1], 
              img_size[1] - new_unpad[0],) # wh padding
    
    dw /= 2 # divide the padding into 2 slides
    dh /= 2

    if (img.shape[1], img.shape[0]) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(144, 144, 114))

    img = (img / 255).astype(np.float32)

    return img


make_calib_man(
    pre_ftn=preprocess_yolo,
    data_dir="calibration_images",             # path to folder of original data files such as images
    save_dir="calibrated_dataset",     # path to folder to save pre-processed calibration data files
    save_name="NIPA_Data_2025_v8",                 # tag name of the calibration dataset   
    max_size=100                             # max number of calibration data files to be saved  
)