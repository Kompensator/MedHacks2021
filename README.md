# CoreSeed: 3D-UNet pipeline with browser-based implementation for Alpha DaRT localization 
# McMedHacks2021
## Ding Yi Zhang, Zara Vajihi, Philippe Marchandise, Jonathan Afilalo 

## Objective:
To automate the localization of Alpha DaRT seeds from CT images using a deep learning model on a browser-based tool for clinical and research use. 
## Introduction: 
Visualization of seed positioning is a critical step to ensure the effectiveness of Alpha DaRT radiation therapy targeted to cancerous cells. 
## Methods: 
We developed the CoreSeed pipeline with the following steps. We crop and resize the region of interest using a heuristic bounding box and contour finding approach. We train and validate (using 20% and 10% of labelled scans, respectively) a 3D-UNet containing 4 levels and 1.3 million parameters to automatically segment the seeds. We add new functionalities to our CoreSlicer browser-based DICOM segmentation tool to handle multi-slice segmentation and integrate the aforementioned model using a Flask endpoint, which in turn, returns the coordinates of the predicted seeds and displays the seed masks overlaying the CT images. 
## Results: 
CoreSeed achieved a DICE score of 93% in the validation set, with the DICE score weighted to be inversely proportional to the number of voxels in each class (to better reflect correct classification of the small seeds rather than the large background class). The pipeline and deep learning model were successfully deployed and accessible on a local version of the CoreSlicer app, allowing the end-user to upload the CT images and run the deep learning model using a graphical interface, then visualize and edit the predicted seed masks in three-dimensions, and export them to a structured output file containing coordinates. 
## Conclusions: 
CoreSeed is an accurate and practical solution to localize and visualize Alpha DaRT seeds for targeted radiation therapy using a 3D-UNet architecture integrated in a user-friendly browser-based web app. 
