import torch
import nibabel as nib
import os
from unet_datasets import *
import numpy as np
from unet_model import UNet3D
from pickle import load
from matplotlib import pyplot as plt
from tqdm import tqdm
from unet_utils import *

def inference():
    with open("UNet3D_config", 'rb') as f:
        model_config = load(f)

    GPU = True

    model = UNet3D(**model_config, testing=True)      # model is running on CPU
    # weights_path = r'C:\Users\dingyi.zhang\Documents\CV-Calcium-DY\checkpoints\dim512_features9_10%_56.h5'      # high DSC: 73%
    # weights_path = r'C:\Users\dingyi.zhang\Documents\CV-Calcium-DY\checkpoints\dim512_features9_20%_90.h5'
    # weights_path = r'C:\Users\dingyi.zhang\Documents\CV-Calcium-DY\checkpoints\dim512_features9_10%_62.h5'      # low CE but really poor performance
    weights_path = r'C:\Users\dingyi.zhang\Documents\CV-Calcium-DY\checkpoints\dim512_DICE+CE_20%_46.h5'
    model.load_state_dict(torch.load(weights_path))
    if GPU:
        model.cuda()

    test_dataset = AlphaTau3_train(start=0.5, end=0.55)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    weight = torch.Tensor([0.01, 1.0])
    # loss = torch.nn.CrossEntropyLoss(weight=weight)
    DICE = GeneralizedDiceLoss(loss=False)

    loss_accum = []
    inputs, outputs, labels = [], [], []
    for i_batch, (data, target) in tqdm(enumerate(test_loader)):
        model.eval()
        with torch.no_grad():
            if GPU:
                data, target = data.cuda(), target.cuda()
            pred = model(data)
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
        inputs.append(data)
        outputs.append(pred)
        labels.append(target)

    # slices_with_ca = []
    # for i in range(target.shape[2]):
    #     if len(np.unique(target[:, :, i])) > 1:
    #         slices_with_ca.append(i)
    #     elif len(np.unique(output[:, :, i])) > 1:
    #         slices_with_ca.append(i)

    print(sum(loss_accum)/(len(loss_accum)))

    save_dir = r'C:\Users\dingyi.zhang\Documents\MedHacks2021\inference'
    for i, (input, output, label) in enumerate(zip(inputs, outputs, labels)):
        nifti = nib.Nifti1Image(input, affine=np.eye(4))
        nib.save(nifti, f'{save_dir}\\{i}_input.nii.gz')
        nifti = nib.Nifti1Image(output, affine=np.eye(4))
        nib.save(nifti, f'{save_dir}\\{i}_output.nii.gz')
        nifti = nib.Nifti1Image(label, affine=np.eye(4))
        nib.save(nifti, f'{save_dir}\\{i}_label.nii.gz')


if __name__ == "__main__":
    inference()