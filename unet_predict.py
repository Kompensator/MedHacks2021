import torch
import nibabel as nib
import os
from unet_datasets import *
import numpy as np
from unet_model import UNet3D
from pickle import load
from matplotlib import pyplot as plt
from tqdm import tqdm

def inference():
    with open("UNet3D_config", 'rb') as f:
        model_config = load(f)
    model = UNet3D(**model_config, testing=True)      # model is running on CPU
    model.load_state_dict(torch.load(r"C:\Users\dingyi.zhang\Documents\CV-Calcium-DY\checkpoints\test_dim512_features9_112.h5"))
    model.cuda()

    test_dataset = AlphaTau3_train(start=0.2, end=0.24)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    weight = torch.Tensor([0.01, 1.0])
    loss = torch.nn.CrossEntropyLoss(weight=weight)

    inputs, outputs, labels = [], [], []
    for i_batch, (data, target) in tqdm(enumerate(test_loader)):
        model.eval()
        with torch.no_grad():
            data, target = data.cuda(), target.cuda()
            pred = model(data)
            pred = torch.argmax(pred, dim=1, keepdim=True)

        pred = pred.cpu().detach().numpy()
        pred = np.squeeze(np.squeeze(pred, axis=0), axis=0)
        data = data.cpu().detach().numpy()
        data = np.squeeze(np.squeeze(data, axis=0), axis=0)
        target = target.cpu().detach().numpy()
        target = np.squeeze(target, axis=0)
        inputs.append(data)
        outputs.append(pred)
        labels.append(target)

    # slices_with_ca = []
    # for i in range(target.shape[2]):
    #     if len(np.unique(target[:, :, i])) > 1:
    #         slices_with_ca.append(i)
    #     elif len(np.unique(output[:, :, i])) > 1:
    #         slices_with_ca.append(i)

    save_dir = r'C:\Users\dingyi.zhang\Documents\MedHacks2021\inference'
    for i, (input, output, label) in enumerate(zip(inputs, outputs, labels)):
        nifti = nib.Nifti1Image(input, affine=np.eye(4))
        nib.save(nifti, f'{save_dir}\\{i}_input.nii.gz')
        nifti = nib.Nifti1Image(output, affine=np.eye(4))
        nib.save(nifti, f'{save_dir}\\{i}_output.nii.gz')
        nifti = nib.Nifti1Image(label, affine=np.eye(4))
        nib.save(nifti, f'{save_dir}\\{i}_label.nii.gz')


    # for i in slices_with_ca:
    #     f, ax = plt.subplots(1, 3)
    #     ax[0].imshow(data[:, :, i], cmap='gray')
    #     ax[0].set_title(f"Slice {i} out of {output.shape[2]}")
    #     ax[1].imshow(output[:, :, i])
    #     ax[1].set_title("3D-UNet prediction")
    #     ax[2].imshow(target[:, :, i])
    #     ax[2].set_title("Ground truth")
    #     plt.show()

if __name__ == "__main__":
    inference()