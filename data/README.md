## All original datasets could be downloaded from:
The retinal Optical Coherence Tomography (OCT) and the Chest X-rays image datasets are available at https://data.mendeley.com/datasets/rscbjbr9sj/2.
The Brain Tumor dataset 1 can be downloaded from https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection.
The Brain Tumor dataset 2 can be found at https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1.


## All data used for training and testing in our paper is available at:
The data used for training and testing in our experiment can be downloaded from https://drive.google.com/drive/folders/1qh8fGZ6aqVSRtWpGHtWdnb6vqRDFUbO\\Y?dmr=1\&ec=wgc-drive-hero-goto


## Explanation Regarding Training and Testing Datasets
- After downloading the datasets, you are supposed to put them into './data/', and the file format reference is as follows. (take the Brain Tumor2 dataset as an example.)

- './data/Brain Tumor2/'
  - trainA_img
    - .png
  - trainB_img
    - .png
  
  - testA_img
    - .png
  - testB_img
    - .png

- training images with normal label will be put into 'trainA_img/' folder, while training images with abnormal labels will be put into 'trainB_img/'
- test images with normal label will be put into 'testA_img/' folder, while test images with abnormal labels will be put into 'testB_img/'
- the names and the labels of the training images (with the format 'image_name label') are put into the 'trainAB_img-name_label.txt'
- the names and the labels of the test images (with the format 'image_name label') are put into the 'testAB_img-name_label.txt'
- '0' represents normal class label while other numbers represent abnormal classes in our work