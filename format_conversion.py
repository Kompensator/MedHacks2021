import numpy as np
import nibabel as nib
import os
from matplotlib import pyplot as plt
import SimpleITK as sitk
import time


nib_dir = r'C:\Users\dingyi.zhang\Documents\MedHacks2021\inference'
files = os.listdir(nib_dir)
os.chdir(nib_dir)

end = 50

for num in range(0, end):
    label_f = '{}_label.nii.gz'.format(num)
    input_f = '{}_input.nii.gz'.format(num)
    output_f = '{}_output.nii.gz'.format(num)

    if os.path.exists(label_f) and os.path.exists(input_f) and os.path.exists(output_f):
        input = np.array(nib.load(input_f).get_fdata())
        label = np.array(nib.load(label_f).get_fdata())
        output = np.array(nib.load(output_f).get_fdata()).astype(np.float64)
        
        # np.save('{}_label'.format(num), label)
        # np.save('{}_output'.format(num), output)
        
        input = np.swapaxes(input,0,2)
        filtered_image = sitk.GetImageFromArray(input)
        series_file_names = [f'output/DCM_{i}.dcm' for i in range(input.shape[2])]

        # Convert numpy array to int
        castFilter = sitk.CastImageFilter()
        castFilter.SetOutputPixelType(sitk.sitkInt16)
        filtered_image = castFilter.Execute(filtered_image)

        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")

        for i in range(filtered_image.GetDepth()): 
            image_slice = filtered_image[:,:,i]
            
            # Set relevant keys indicating the change, modify or remove private tags as needed
            image_slice.SetMetaData("0008|0031", modification_time)
            image_slice.SetMetaData("0008|0021", modification_date)
            image_slice.SetMetaData("0008|0008", "ALPHATAU3\{}".format(num))
            
            # Each of the UID components is a number (cannot start with zero) and separated by a '.'
            # We create a unique series ID using the date and time.
            image_slice.SetMetaData("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time)
            
            # Write to the output directory and add the extension dcm if not there, to force writing is in DICOM format.
            writer.SetFileName( series_file_names[i] )
            writer.Execute(image_slice)
        break
