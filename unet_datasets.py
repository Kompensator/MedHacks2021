
import random
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import os
import torch
from termcolor import cprint
import cv2
from scipy.ndimage import zoom
from tqdm import tqdm
from torchvision.transforms import Normalize
import nrrd
from multiprocessing import Pool
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
    def __init__(self, start=0.0, end=0.8, seed=42):
        super(AlphaTau3_train, self).__init__()
        random.seed(seed)
        self.start = start
        self.end = end
        self.data_path = r'C:\Users\dingyi.zhang\Documents\MedHacks2021\Tau3_train'
        self.data_path_list = os.listdir(self.data_path)
        self.data_path_list.sort()
        self.image_list = []
        self.label_list = []
        self.images = []
        self.labels = []
        for i in self.data_path_list:
            if "image" in i:
                self.image_list.append(i)
            elif "mask" in i:
                self.label_list.append(i)

        c = list(zip(self.image_list, self.label_list))
        random.shuffle(c)
        self.image_list, self.label_list = zip(*c)

        self.image_list = self.image_list[int(start*len(self.image_list)): int(end*len(self.image_list))]
        self.label_list = self.label_list[int(start*len(self.label_list)): int(end*len(self.label_list))]

        for image_name, label_name in zip(self.image_list, self.label_list):
            if image_name.split('_')[0] == label_name.split('_')[0]:
                try:
                    image = np.array(nib.load(os.path.join(self.data_path, image_name)).get_fdata())
                    label = np.array(nib.load(os.path.join(self.data_path, label_name)).get_fdata())
                    assert image.shape == label.shape, "{} and {} dont have same shape".format(image_name, label_name)

                    self.images.append(np.expand_dims(image, axis=0))
                    self.labels.append(label)
                except:
                    cprint("Error while loading {}".format(image_name))
                    continue
            else:
                cprint(f"{image_name} and {label_name} do not match", "red")
        
        self.images = torch.from_numpy(np.array(self.images)).float()
        self.labels = torch.from_numpy(np.array(self.labels)).long()

        #TODO normalization

class CVC(Dataset):
    def __init__(self, start=0.0, end=0.8, size=(224, 224, 100), seed=42, cores=10, redo=False):
        super(CVC, self).__init__()
        random.seed(seed)
        self.size = size
        self.resize_cache = '.\\resize_cache\\'

        self.image_dir = r'C:\Users\dingyi.zhang\Dropbox\CVC data\raw'
        self.image_paths = []
        self.label_paths = []
        self.images = []
        self.labels = []

        """ NOTE assumes this structure:
        raw/
            rvh-001/
                    img.nrrd
                    label.nii.gz
            bid-002/
                    img.nrrd
                    label.nii.gz
        """
        folders = os.listdir(self.image_dir)
        for f in folders:
            files = os.listdir(os.path.join(self.image_dir, f))
            img_present, label_present = False, False
            for _f in files:
                if 'img' in _f:
                    temp_img = os.path.join(self.image_dir, f, _f)
                    img_present = True
                elif 'label' in _f:
                    temp_label = os.path.join(self.image_dir, f, _f)
                    label_present = True
            if img_present and label_present:
                self.image_paths.append(temp_img)
                self.label_paths.append(temp_label)
            else:
                cprint(f"{f} folder skipped, either img or label not there", 'red')
        
        assert len(self.image_paths) == len(self.label_paths), "Img and label files don't match"

        # shuffeling images coupled with labels
        c = list(zip(self.image_paths, self.label_paths))
        random.shuffle(c)
        self.image_paths, self.label_paths = zip(*c)

        start = int(start * len(self.image_paths))
        end = int(end * len(self.image_paths))
        
        # self.image_paths = [r'C:\Users\dingyi.zhang\Dropbox\CVC data\raw\Rvh-137-Gv\img.nrrd']
        # self.label_paths = [r'C:\Users\dingyi.zhang\Dropbox\CVC data\raw\Rvh-137-Gv\label.nii.gz']
        
        for d1, d2 in zip(self.image_paths[start: end], self.label_paths[start: end]):
            cropping = [0.300, 0.879, 0.195, 0.684, 0.0625, 0.80]

            arr1 = self.preprocess(d1, "linear", cropping=cropping, redo=redo)
            arr2 = self.preprocess(d2, "nearest", cropping=cropping, redo=redo)
            assert arr1.shape == arr2.shape, f"resized {d1} and {d2} shape do not match"

            self.images.append(np.expand_dims(arr1, axis=0))
            self.labels.append(arr2)            # NOTE nn.CrossEntropy does not require extra C dim
        
        self.images = torch.from_numpy(np.array(self.images)).float()
        self.labels = torch.from_numpy(np.array(self.labels)).long()
        img_mean = torch.mean(self.images)
        img_std = torch.std(self.images)
        self.img_normalize = Normalize([img_mean], [img_std])
        self.images = self.img_normalize(self.images)

        print("Image size: {}\tLabel size: {}".format(self.images.size(), self.labels.size()))

    def preprocess(self, arr_name, interpolation, cropping=None, redo=False):
        """ Idea: 3D zooming is very slow, so we save the resized image to a file, and load it back if exists
            Add all attributes of the final image i.e. transforms and cropping to fname 
        """
        fname = "{}_{}_{}x{}x{}".format(arr_name.split("\\")[-2], interpolation, self.size[0], self.size[1], self.size[2])
        if not redo and os.path.exists(os.path.join(self.resize_cache, fname + '.npy')):
            cprint("Loading cached resize {}".format(fname), 'yellow')
            return np.load(os.path.join(self.resize_cache, fname + '.npy'))
        else:
            cprint("Loading and resizing {}, saving to cache".format(fname), 'green')
            if '.nii' in arr_name:
                arr = np.array(nib.load(arr_name).get_fdata())
            elif '.nrrd' in arr_name:
                arr = nrrd.read(arr_name)[0]
            else:
                cprint(f"{arr_name} has unrecognized ext", "red")

            xlim = [cropping[0], cropping[1]]
            ylim = [cropping[2], cropping[3]]
            zlim = [cropping[4], cropping[5]]

            arr = self.heuristical_crop(arr_name, arr, xlim, ylim, zlim)
            exceptions = ['rvh', 'bid-004', 'bid-006']
            for e in exceptions:
                if e in arr_name.lower():
                # RVH scans are somehow mirrored in coronal plane, looks like dextrocardia
                    arr = arr[::-1, ::-1, :]

            # nifti = nib.Nifti1Image(arr, np.eye(4))
            # nib.save(nifti, 'img_before.nii.gz')
    
            scaling = (self.size[0]/arr.shape[0], self.size[1]/arr.shape[1], self.size[2]/arr.shape[2])
            arr = self.grid_interpolator(arr, scaling, interpolation)
            # nifti = nib.Nifti1Image(arr, np.eye(4))
            # nib.save(nifti, 'img_after.nii.gz')
            np.save(os.path.join(self.resize_cache, fname), arr)
            return arr

    def heuristical_crop(self, arr_name, arr, xlim, ylim, zlim):
        """ TODO think about diff cropping for RVH and BID"""
        xlim[0] = int(xlim[0] * arr.shape[0])
        xlim[1] = int(xlim[1] * arr.shape[0])
        ylim[0] = int(ylim[0] * arr.shape[1])
        ylim[1] = int(ylim[1] * arr.shape[1])
        zlim[0] = int(zlim[0] * arr.shape[2])
        zlim[1] = int(zlim[1] * arr.shape[2])
        print(f"{arr_name} before crop: {arr.shape}")
        arr = arr[xlim[0]:xlim[1], ylim[0]:ylim[1], zlim[0]:zlim[1]]
        print(f"{arr_name} after crop: {arr.shape}")
        return arr

    def grid_interpolator(self, arr, scaling, interpolation):
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

    def worker(self, dirs):
        d1 = dirs[0]
        d2 = dirs[1]
        path = os.path.join(self.image_dir, d1, 'img.nrrd')
        arr1 = nrrd.read(path)[0]            # shape = (x, y, z)

        path = os.path.join(self.label_dir, d2)
        arr2 = np.array(nib.load(path).get_fdata())
        assert arr1.shape == arr2.shape, "image and label shape not match"
        
        scaling = (self.size[0]/arr1.shape[0], self.size[1]/arr1.shape[1], self.size[2]/arr1.shape[2])
        arr1 = zoom(arr1, scaling)
        arr2 = zoom(arr2, scaling)
        self.images.append(np.expand_dims(arr1, axis=0))
        self.labels.append(arr2)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        return self.images[idx], self.labels[idx]


class Custom_dataset(Dataset):
    """ NO LONGER USED 
    Very rudimentary workaround for the complex complicated convoluted dataloading system.
    __init__ loads data and convert into tensor
    __getitem__ returns the actual tensor data at each epoch
    """
    def __init__(self, img_path, gt_path):
        self.gt_path = gt_path
        self.img_path = img_path
        self.gt_list = os.listdir(self.gt_path)
        self.img_list = os.listdir(self.img_path)
        self.images, self.gt = [], []
        for d in self.img_list:
            x = np.load(os.path.join(self.img_path, d))
            self.images.append(np.expand_dims(x, axis=0))
        for d in self.gt_list:
            y = np.load(os.path.join(self.gt_path, d))
            self.gt.append(np.expand_dims(y, axis=0))
        self.images = torch.from_numpy(np.array(self.images)).float()
        self.gt = torch.from_numpy(np.array(self.gt)).float()
        self.batch_size = 5
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        curr_img = self.images[idx]
        curr_gt = self.gt[idx]
        sample = {'image': curr_img, 'label': curr_gt}
        # return sample
        return curr_img, curr_gt

    def augment(self):
        pass


class Pancreas_dataset(Dataset):
    def __init__(self, img_path, gt_path, transforms=None, n=56):
        self.n = n
        self.img_path = img_path
        self.gt_path = gt_path
        self.gt_list = os.listdir(self.gt_path)
        self.img_list = os.listdir(self.img_path)
        self.img_taken = []
        self.gt_taken = []
        self.transforms = transforms
        self.images, self.gt = [], []
        target_dim = (200, 200, 100)

        # loading from image0001 to image00(n)
        i = 0
        for d in tqdm(self.img_list):
            if "image" in d and ".nii.gz" in d and i < self.n:
                self.img_taken.append(d)
                x = np.array(nib.load(os.path.join(self.img_path, d)).get_fdata())
                scaling = (target_dim[0]/x.shape[0], target_dim[1]/x.shape[1], target_dim[2]/x.shape[2])
                x = zoom(x, scaling)
                self.images.append(np.expand_dims(x, axis=0))
                i += 1

        i = 0
        for d in tqdm(self.gt_list):
            if "label" in d and ".nii.gz" in d and i < self.n:
                self.gt_taken.append(d)
                x = np.array(nib.load(os.path.join(self.gt_path, d)).get_fdata(), dtype=np.int16)
                scaling = (target_dim[0]/x.shape[0], target_dim[1]/x.shape[1], target_dim[2]/x.shape[2])
                x = zoom(x, scaling)
                self.gt.append(np.expand_dims(x, axis=0))
                i += 1
        
        for i, j in zip(self.img_taken, self.gt_taken):
            assert i[5:9] == j[5:9], f"Image and GT data don't concur ({i} vs {j}"

        self.images = torch.from_numpy(np.array(self.images)).float()
        self.gt = torch.from_numpy(np.array(self.gt)).float()

        # standardize CT image data with mean = 0, std = 1
        # skip normalization for GT bcz tensors are either 0. or 1.
        img_mean = torch.mean(self.images)
        img_std = torch.std(self.images)
        self.img_normalize = Normalize([img_mean], [img_std])
        self.images = self.img_normalize(self.images)

        assert self.images.shape == self.gt.shape, f"Somehow images and GT don't have same shape: {self.images.shape} vs {self.gt.shape}"
        cprint("Training dataset shape: {}".format(self.gt.shape), 'red')

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        curr_img = self.images[idx]
        curr_gt = self.gt[idx]
        # apply augmentation transforms
        if self.transforms is not None:
            for t in self.transforms:
                curr_img = t(curr_img)
                curr_gt = t(curr_gt)
        sample = {'image': curr_img, 'label': curr_gt}
        # return sample
        return curr_img, curr_gt


class Pancreas_val(Pancreas_dataset):
    def __init__(self, img_path, gt_path, transforms=None, n=12):
        self.img_path = img_path
        self.gt_path = gt_path
        self.images, self.gt = [], []
        target_dim = (200, 200, 100)
        self.transforms = transforms
        
        start = 70
        end = start + n + 1
        assert end <= 83, "n is set too high for validation set!"

        # loading from image0070 to image0082
        for i in tqdm(range(start, end)):
            os.chdir(self.img_path)
            img_exists = os.path.isfile(f"image00{i}.nii.gz")
            os.chdir(self.gt_path)
            gt_exists = os.path.isfile(f"label00{i}.nii.gz")
            if img_exists and gt_exists:
                x = np.array(nib.load(os.path.join(self.img_path, f"image00{i}.nii.gz")).get_fdata())
                scaling = (target_dim[0]/x.shape[0], target_dim[1]/x.shape[1], target_dim[2]/x.shape[2])
                x = zoom(x, scaling)
                self.images.append(np.expand_dims(x, axis=0))

                y = np.array(nib.load(os.path.join(self.gt_path, f"label00{i}.nii.gz")).get_fdata(), dtype=np.int16)
                scaling = (target_dim[0]/y.shape[0], target_dim[1]/y.shape[1], target_dim[2]/y.shape[2])
                y = zoom(y, scaling)
                self.gt.append(np.expand_dims(y, axis=0))
    
        self.images = torch.from_numpy(np.array(self.images)).float()
        self.gt = torch.from_numpy(np.array(self.gt)).float()

        img_mean = torch.mean(self.images)
        img_std = torch.std(self.images)
        self.img_normalize = Normalize([img_mean], [img_std])
        self.images = self.img_normalize(self.images)

        cprint("Validation dataset shape: {}".format(self.gt.shape), 'red')


if __name__ == "__main__":
    a = AlphaTau3_train(start=0.0, end=0.01)