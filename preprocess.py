""" last edited by: DY, 11:34
"""

import numpy as np
from matplotlib import pyplot as plt
import nibabel as nib
import os
import cv2
from termcolor import cprint
from copy import deepcopy
from tqdm import tqdm
from multiprocessing import Pool

tau3_hard = r'C:\Users\dingyi.zhang\Downloads\AlphaTau3\hard_dataset'
tau1 = r'C:\Users\dingyi.zhang\Documents\AlphaTau\Tau1'
tau3_train = r'C:\Users\dingyi.zhang\Downloads\AlphaTau3\train'

current_folder = tau3_train

padding = 20
square = True       # if False, bounding box will be a rectangle tangential to the outline of body

scans = os.listdir(current_folder)


def worker(s):
    img_dir = os.path.join(current_folder, s + f'\\images\\{s}.npy')
    mask_dir = os.path.join(current_folder, s + '\\masks\\mask.npy')
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
            # contour_plot = cv2.drawContours(slice, [contour], -1, 1, 1)
            # plt.imshow(contour_plot)
            # plt.show()
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
        X_MIN, X_MAX, Y_MIN, Y_MAX = X_MIN-padding, X_MAX+padding, Y_MIN-padding, Y_MAX+padding
        x_center = (X_MAX + X_MIN) / 2
        y_center = (Y_MAX + Y_MIN) / 2
        x_range = X_MAX - X_MIN
        y_range = Y_MAX - Y_MIN

        # make the bounding box a square based on the largest dimension
        if square:
            if x_range > y_range:
                Y_MAX = int(y_center + x_range/2)
                Y_MIN = int(y_center - x_range/2)
            elif y_range > x_range:
                X_MAX = int(x_center + y_range/2)
                X_MIN = int(x_center - y_range/2)

        print(f"{s} after crop: {X_MAX-X_MIN}x{Y_MAX-Y_MIN}")

        # plt.imshow(slice)
        # plt.plot(X_MIN, Y_MIN, 'r.')
        # plt.plot(X_MAX, Y_MAX, 'r.')
        # plt.show()

        # crop the image
        img = orig_img[X_MIN:X_MAX, Y_MIN:Y_MAX, :]
        mask = mask[X_MIN:X_MAX, Y_MIN:Y_MAX, :]

        save_dir = r'C:\Users\dingyi.zhang\Documents\MedHacks2021\Tau3_train'
        nifti = nib.Nifti1Image(img, np.eye(4))
        nib.save(nifti, f"{save_dir}\\{s}_image.nii.gz")
        nifti = nib.Nifti1Image(mask, np.eye(4))
        nib.save(nifti, f"{save_dir}\\{s}_mask.nii.gz")
    
if __name__ == "__main__":
    # multiprocessing
    p = Pool(10)
    p.map(worker, scans)

    # simple processing
    # for s in tqdm(scans):
    #     worker(s)