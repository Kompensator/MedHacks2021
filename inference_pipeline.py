import os
from sys import argv
import numpy as np
from termcolor import cprint
from copy import deepcopy
import cv2
import nibabel as nib
from unet_datasets import *
from unet_model import *
from unet_utils import *
from pickle import load
from tqdm import tqdm

""" Runs the entire segmentation inference pipeline
    - takes in a the name of folder (ex '001')
    - outputs the segmentation in npy
"""

def worker(s, current_folder, square=True, padding=20):
    img_dir = os.path.join(current_folder, s + f'\\images\\{s}.npy')
    mask_dir = os.path.join(current_folder, s + '\\masks\\mask.npy')
    img = np.load(img_dir)
    orig_shape = img.shape
    orig_img = deepcopy(img)

    has_mask = True
    try:
        mask = np.load(mask_dir)
    except:
        mask = np.zeros_like(img)
        has_mask = False
    
    orig_mask = deepcopy(mask)
    
    assert img.shape == mask.shape, f"somehow {s} doens't have same shape img vs mask"
    if img.shape[0] != img.shape[1]:
        cprint(f"{s} is non square, this means the data is likely incomplete", 'red')
        cprint(f"{s} shape: {img.shape}", 'red')
        complete_data = True
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

        # make the bounding box a square as much as possible
        if square:
            if x_range > y_range:
                Y_MAX = int(y_center + x_range/2)
                Y_MIN = int(y_center - x_range/2)
            elif y_range > x_range:
                X_MAX = int(x_center + y_range/2)
                X_MIN = int(x_center - y_range/2)

            # Correct bug where box is out of bounds
            if X_MIN < 0:
                X_MIN = 0
            if X_MAX > img.shape[0]:
                X_MAX = img.shape[0]
            if Y_MIN < 0:
                Y_MIN = 0
            if Y_MAX > img.shape[1]:
                Y_MAX = img.shape[1]

        print(f"{s} after crop: {X_MAX-X_MIN}x{Y_MAX-Y_MIN}")

        # plt.imshow(slice)
        # plt.plot(X_MIN, Y_MIN, 'r.')
        # plt.plot(X_MAX, Y_MAX, 'r.')
        # plt.show()

        # crop the image
        img = orig_img[X_MIN:X_MAX, Y_MIN:Y_MAX, :]
        mask = mask[X_MIN:X_MAX, Y_MIN:Y_MAX, :]
        save_dir = r'C:\Users\dingyi.zhang\Documents\MedHacks2021\temp'     # temp save dir for inference
        nifti = nib.Nifti1Image(img, np.eye(4))
        nib.save(nifti, f"{save_dir}\\{s}_image.nii.gz")
        nifti = nib.Nifti1Image(mask, np.eye(4))
        nib.save(nifti, f"{save_dir}\\{s}_mask.nii.gz")

        return orig_img, orig_mask, has_mask, orig_shape, [X_MIN, X_MAX, Y_MIN, Y_MAX, 0, img.shape[2]]

def inference(test_loader, has_mask, weights_path, GPU=False):
    with open("UNet3D_config_384_18", 'rb') as f:
        model_config = load(f)

    model = UNet3D(**model_config, testing=True)      # model is running on CPU
    model.load_state_dict(torch.load(weights_path))
    if GPU:
        model.cuda()

    weight = torch.Tensor([0.01, 1.0])
    # loss = torch.nn.CrossEntropyLoss(weight=weight)
    DICE = GeneralizedDiceLoss(loss=False, GPU=GPU)

    loss_accum = []
    inputs, outputs, labels = [], [], []
    for i_batch, (data, target) in tqdm(enumerate(test_loader)):
        model.eval()
        with torch.no_grad():
            if GPU:
                data, target = data.cuda(), target.cuda()
            pred = model(data)
            if has_mask:
                loss = DICE(pred, target)
                loss_accum.append(loss.item())
            pred = torch.argmax(pred, dim=1, keepdim=True)

        if GPU:
            pred = pred.cpu().detach().numpy()
            data = data.cpu().detach().numpy()
            target = target.cpu().detach().numpy()
        else:
            pred = pred.numpy()
            data = data.numpy()
            target = target.numpy()
        pred = np.squeeze(np.squeeze(pred, axis=0), axis=0)
        data = np.squeeze(np.squeeze(data, axis=0), axis=0)
        target = np.squeeze(target, axis=0)
        # inputs.append(data)
        outputs.append(pred)
        # labels.append(target)
    
    if has_mask:
        return outputs, loss_accum
    else:
        return outputs, None

def pad(arr, target_shape):
    """ In case arr.shape != target_shape due to interpolation"""
    diff_shape = [i-j for i, j in zip(target_shape, arr.shape)]
    arr = np.pad(arr, ((diff_shape[0], 0), (diff_shape[1], 0), (0, diff_shape[2])), 'constant')
    return arr

def main(folders):
    padding = 20
    square = True       # if False, bounding box will be a rectangle tangential to the outline of body

    # temp_files = os.listdir(r'C:\Users\dingyi.zhang\Documents\MedHacks2021\temp')
    # for file in temp_files:
    #     os.remove(r'C:\Users\dingyi.zhang\Documents\MedHacks2021\temp\\' + file)

    dataset = r'C:\Users\dingyi.zhang\Downloads\AlphaTau3\test'        # unzipped alphatau dataset
    imgs, masks, has_masks, orig_sizes, crops = [], [], [], [], []
    for folder in folders:
        img, mask, has_mask, orig_size, crop = worker(folder, dataset)
        imgs.append(img)
        # masks.append(mask)
        has_masks.append(has_mask)
        orig_sizes.append(orig_size)
        crops.append(crop)

    after_crop_sizes = []
    for crop in crops:
        after_crop_size = (crop[1] - crop[0], crop[3] - crop[2], crop[5] - crop[4])
        after_crop_sizes.append(after_crop_size)
    NN_input_size = (384, 384, 40)

    test_set = AlphaTau3_train(start=0.0, end=1.0, data_path= r'C:\Users\dingyi.zhang\Documents\MedHacks2021\temp', cores=10, rand=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)

    # inference
    outputs, losses = inference(test_loader, has_mask, weights_path=r'C:\Users\dingyi.zhang\Documents\CV-Calcium-DY\checkpoints\dim384_18_LR0.0001_110.h5', GPU=False)

    for i, output in enumerate(outputs):
        # undo the resizing from cropped to NN input
        reverse_scaling = [after_crop_sizes[i][0]/NN_input_size[0], after_crop_sizes[i][1]/NN_input_size[1], after_crop_size[i][2]/NN_input_size[2]]
        output = grid_interpolator(output, reverse_scaling, "nearest")

        # (usually Z is off by 1)
        output = pad(output, after_crop_sizes[i])
        
        # re-crop to original size
        output = np.pad(output, ((crops[i][0], orig_sizes[i][0]-crops[i][1]), (crops[i][2], orig_sizes[i][1]-crops[i][3]), (0, 0)), 'constant', constant_values=0)

        assert output.shape == imgs[i].shape, "Final output shape is not the same as input shape!!"

        np.save('test{}_predicted'.format(folders[i]), output)

        nifti = nib.Nifti1Image(output, np.eye(4))
        nib.save(nifti, 'test{}_predicted.nii.gz'.format(folders[i]))
        # nifti = nib.Nifti1Image(masks[i], np.eye(4))
        # nib.save(nifti, '{}_label.nii.gz'.format(folders[i]))
        nifti = nib.Nifti1Image(imgs[i], np.eye(4))
        nib.save(nifti, 'test{}_input.nii.gz'.format(folders[i]))

    # saving img as dicom logic here

if __name__ == "__main__":
    # if len(argv) < 2:
    #     print("No file provided, running 014 instead")
    #     folder = ['014']
    # else:
    #     folder = [argv[1]]

    root = r'C:\Users\dingyi.zhang\Downloads\AlphaTau3\test'
    folders = os.listdir(root)
    main(folders)