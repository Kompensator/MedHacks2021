
import random
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import os
import torch
from termcolor import cprint
import cv2
from scipy.ndimage import zoom
from tqdm import tqdm
from torchvision.transforms import Normalize
import nrrd
from multiprocessing import Pool, current_process
from scipy.interpolate import RegularGridInterpolator
from matplotlib import pyplot as plt


def random_flip(m, random_state, axis_prob=0.5, **kwargs):
    assert random_state is not None, "Random state cannot be none"
    for axis in (0, 1, 2):
        if random_state.uniform() > axis_prob:
            m = np.flip(m, axis)
    return m

def random_rotate_90(m, random_state, **kwargs):
    assert random_state is not None, "Random state cannot be none"
    k = random_state.randint(0, 4)
    axis = (1, 2)
    if m.ndim == 3:
        m = np.rot90(m, k, axis)
    else:
        channels = [np.rot90(m[c], k, axis) for c in range(m.shape[0])]
        m = np.stack(channels, axis=0)
    return m

class AlphaTau3_train(Dataset):
    def __init__(self, start=0.0, end=0.8, seed=42, data_path=r'C:\Users\dingyi.zhang\Documents\MedHacks2021\Tau3_train', cores=10):
        super(AlphaTau3_train, self).__init__()
        self.images, self.labels = boss(start, end, seed, data_path, cores)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def boss(start=0.0, end=0.8, seed=42, data_path=r'C:\Users\dingyi.zhang\Documents\MedHacks2021\Tau3_train', cores=10):
    random.seed(seed)
    start = start
    end = end
    data_path_list = os.listdir(data_path)
    data_path_list.sort()
    image_list = []
    label_list = []
    images = []
    labels = []
    for i in data_path_list:
        if "image" in i:
            image_list.append(i)
        elif "mask" in i:
            label_list.append(i)

    c = list(zip(image_list, label_list))
    random.shuffle(c)
    image_list, label_list = zip(*c)

    image_list = image_list[int(start*len(image_list)): int(end*len(image_list))]
    label_list = label_list[int(start*len(label_list)): int(end*len(label_list))]

    jobs = [[data_path, image_path, label_path] for image_path, label_path in zip(image_list, label_list)]
    p = Pool(cores)
    rtrn = p.map_async(worker, jobs)

    loaded = rtrn.get()

    images = [job[0] for job in loaded if job is not None]
    labels = [job[1] for job in loaded if job is not None]
    images = torch.from_numpy(np.array(images)).float()
    labels = torch.from_numpy(np.array(labels)).long()

    img_mean = torch.mean(images)
    img_std = torch.std(images)
    img_normalize = Normalize([img_mean], [img_std])
    images = img_normalize(images)
    return images, labels

def worker(job):
    data_path = job[0]
    image_name = job[1]
    label_name = job[2]
    if image_name.split('_')[0] == label_name.split('_')[0]:
        try:
            image = np.array(nib.load(os.path.join(data_path, image_name)).get_fdata())
            label = np.array(nib.load(os.path.join(data_path, label_name)).get_fdata())
            assert image.shape == label.shape, "{} and {} dont have same shape".format(image_name, label_name)

            target_dim = (512, 512, 40)
            scaling = (target_dim[0]/image.shape[0], target_dim[1]/image.shape[1], target_dim[2]/image.shape[2])
            image = grid_interpolator(image, scaling, 'linear')
            label = grid_interpolator(label, scaling, 'nearest')
            
            assert image.shape == label.shape, "{} and {} dont have same shape after resizing".format(image_name, label_name)
            
            # one-hot encode all class in label to be 1
            label[label != 0] = 1

            if image.shape != target_dim:
                image = pad(image, target_dim)
                label = pad(label, target_dim)
            cprint('Process {}: {} loaded successfully!'.format(current_process().name, image_name), 'green')
            return [np.expand_dims(image, axis=0), label]
        except:
            cprint("Process {}: Error while loading {}".format(current_process().name, image_name), 'red')
            return None

    else:
        cprint(f"Process {current_process().name}: {image_name} and {label_name} do not match", "red")
        return None

def pad(arr, target_shape):
    """ In case arr.shape != target_shape due to interpolation"""
    diff_shape = [i-j for i, j in zip(target_shape, arr.shape)]
    arr = np.pad(arr, ((0, diff_shape[0]), (0, diff_shape[1]), (0, diff_shape[2])), 'constant')
    return arr
    
def grid_interpolator(arr, scaling, interpolation):
    """ Implements this solution to resize in 3D using linear for scan and nearest for label
        https://stackoverflow.com/questions/47775621/interpolate-resize-3d-array    
    """
    orig_shape  = arr.shape
    target_shape = (scaling[0]*orig_shape[0], scaling[1]*orig_shape[1], scaling[2]*orig_shape[2])

    steps = [1.0, 1.0, 1.0]
    x, y, z = [steps[k] * np.arange(arr.shape[k]) for k in range(3)]  # original grid
    interpolator = RegularGridInterpolator((x, y, z), arr, method=interpolation)

    new_steps = [steps[i] * (orig_shape[i] / target_shape[i]) for i in range(3)]
    dx, dy, dz = new_steps[0], new_steps[1], new_steps[2]
    new_grid = np.mgrid[0:x[-1]:dx, 0:y[-1]:dy, 0:z[-1]:dz]   # new grid
    new_grid = np.moveaxis(new_grid, (0, 1, 2, 3), (3, 0, 1, 2))  # reorder axes for evaluation
    
    return interpolator(new_grid)

if __name__ == "__main__":
    dataset = AlphaTau3_train(start=0.0, end=0.01)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, (image, label) in enumerate(loader):
        nifti = nib.Nifti1Image(image.numpy()[0, 0, :, :, :], np.eye(4))
        nib.save(nifti, os.path.join("./", "image_{}.nii.gz".format(i)))
        nifti = nib.Nifti1Image(label.numpy()[0, :, :, :], np.eye(4))
        nib.save(nifti, os.path.join("./", "label_{}.nii.gz".format(i)))