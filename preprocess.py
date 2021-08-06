""" last edited by: DY, 11:34
"""

import numpy as np
from matplotlib import pyplot as plt
import nibabel as nib
import os
import cv2
from termcolor import cprint
from copy import deepcopy

tau3_hard = r'C:\Users\dingyi.zhang\Downloads\AlphaTau3\hard_dataset'
tau1 = r'C:\Users\dingyi.zhang\Documents\AlphaTau\Tau1'
os.chdir(r"C:\Users\dingyi.zhang\Documents\AlphaTau")

scans = os.listdir(tau3_hard)

def bbox2(img, padding=40):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return ymin-padding, ymax+padding, xmin-padding, xmax+padding


for s in scans:
    img_dir = os.path.join(tau3_hard, s + f'\\images\\{s}.npy')
    mask_dir = os.path.join(tau3_hard, s + '\\masks\\mask.npy')
    img = np.load(img_dir)
    mask = np.load(mask_dir)
    
    assert img.shape == mask.shape,f"somehow {s} doens't have same shape img vs mask"
    if img.shape[0] != img.shape[1]:
        cprint(f"{s} is non square, this means the data is likely incomplete", 'red')
        cprint(f"{s} shape: {img.shape}", 'red')
        complete_data = False
    else:
        cprint(f"{s} shape: {img.shape}", 'green')
        complete_data = True
    
    if complete_data:
        # first threshold the image to exclude everything with neg intensities
        orig_img = deepcopy(img)
        img[img < -80] = 0
        img[img > 175] = 0
        img[img != 0] = 1

        X_MIN, X_MAX, Y_MIN, Y_MAX = 5e3, 0, 5e3, 0
        for i in range(img.shape[2]):
            slice = img[:,:,i].astype(np.uint8)
            contour, hierarchy = cv2.findContours(slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = np.array(contour)
            longest = 0
            for i, c in enumerate(contour):
                if len(c) > longest:
                    longest = i
            contour = contour[longest]
            contour = np.squeeze(contour, axis=1)

            # find bounding cooredinates
            xmin, xmax, ymin, ymax = 5e3, 0, 5e3, 0
            for c in contour:
                if c[0] < xmin:
                    xmin = c[0]
                if c[0] > xmax:
                    xmax = c[0]
                if c[1] < ymin:
                    ymin = c[1]
                if c[1] > ymax:
                    ymax = c[1]
            
            # update global bounding box of entire volume
            if xmin < X_MIN:
                X_MIN = xmin
            if xmax > X_MAX:
                X_MAX = xmax
            if ymin < Y_MIN:
                Y_MIN = ymin
            if ymax > Y_MAX:
                Y_MAX = ymax
        
        # add padding to bounding box
        padding = 15
        X_MIN, X_MAX, Y_MIN, Y_MAX = X_MIN-padding, X_MAX+padding, Y_MIN-padding, Y_MAX+padding
        print(f"Bounding box for {s}: {X_MIN}, {X_MAX}, {Y_MIN}, {Y_MAX}")

        # crop the image
        img = orig_img[X_MIN:X_MAX, Y_MIN:Y_MAX, :]
        mask = mask[X_MIN:X_MAX, Y_MIN:Y_MAX, :]

        nifti = nib.Nifti1Image(img, np.eye(4))
        nib.save(nifti, f"{s}_image.nii.gz")
        nifti = nib.Nifti1Image(mask, np.eye(4))
        nib.save(nifti, f"{s}_mask.nii.gz")
    
