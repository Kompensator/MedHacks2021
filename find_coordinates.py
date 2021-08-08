# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 08:35:39 2021

@author: zvaj2620
"""
import numpy as np
import cc3d
import nibabel as nib
import os
from termcolor import cprint

def degrees_of_cube(input_matrix):
    dims = np.where(input_matrix!=0)
    
    min_index_x = min(dims[0])
    max_index_x = max(dims[0])
    min_index_y = min(dims[1])
    max_index_y = max(dims[1])
    min_index_z = min(dims[2])
    max_index_z = max(dims[2]) 
    
    dx_min = dims[1][0]
    dx_max = dims[1][-1]
    dy_min = dims[0][0]
    dy_max = dims[0][-1] 
    dz_min = dims[2][0]
    dz_max = dims[2][-1] 
    
    dx = dx_max - dx_min
    dy = dy_max - dy_min
    dz = dz_max - dz_min
    
    center = ((min_index_x+max_index_x)/2, (min_index_y+max_index_y)/2, (min_index_z+max_index_z)/2)

    return center, (dx, dy, dz)


test_masks_dir = r'C:\Users\dingyi.zhang\Documents\MedHacks2021\test_masks'
os.chdir(test_masks_dir)
test_masks = os.listdir(test_masks_dir)
for t in test_masks:
    path_predicted_mask = 'C:\\Users\\dingyi.zhang\\Documents\\MedHacks2021\\test_masks\\' + t
    if t.endswith('.nii.gz'):
        predicted_mask = nib.load(path_predicted_mask)
    else:
        continue
    predicted_mask = predicted_mask.get_fdata()

    predicted_mask = np.array(predicted_mask, dtype=np.uint64)
    labels_out, N = cc3d.connected_components(predicted_mask, return_N=True)

    slices = []

    for segid in range(1, N+1): 
        # print('------------------------',segid,'------------------------')
        extracted_image = (labels_out == segid)
        center , (dx, dy, dz)= degrees_of_cube(extracted_image)
        sq = np.sqrt(np.square(dx) + np.square(dy) + np.square(dz))
        dx = dx/sq
        dy = dy/sq
        dz = dz/sq
        # print(center , (dx, dy, dz)/sq)

        slice = []
        slice.append(center)
        slice.append([dx, dy, dz])
        slices.append(slice)
    
    arr = np.array(slices)
    cprint(t, 'green')
    cprint(arr.shape, 'green')

    np.save("result_{}".format(t.split('_')[0][4:]), arr)

