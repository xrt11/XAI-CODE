# CAE
This is the official code repository for "Accurate Explanation Model for Image Classifiers using Class Association Embedding", which has been published on IEEE 40th International Conference on Data Engineering (ICDE 2024). The paper of this work can be downloaded from https://arxiv.org/abs/2406.07961 or https://ieeexplore.ieee.org/document/10597966.
We advise using the paper version on arxiv.org for reading because the version published on IEEE Xplore is compressed and the figures in this version are unclear. 


## Abstract
Image classification is a primary task in data analysis where explainable models are crucially demanded in various applications. Although amounts of methods have been proposed to obtain explainable knowledge from the black-box classifiers, these approaches lack the efficiency of extracting global knowledge regarding the classification task, thus is vulnerable to local traps and often leads to poor accuracy. In this study, we propose a generative explanation model that combines the advantages of global and local knowledge for explaining image classifiers. We develop a representation learning method called class association embedding (CAE), which encodes each sample into a pair of separated class-associated and individual codes. Recombining the individual code of a given sample with altered class-associated code leads to a synthetic real-looking sample with preserved individual characters but modified class-associated features and possibly flipped class assignments. A building-block coherency feature extraction algorithm is proposed that efficiently separates class-associated features from individual ones. The extracted feature space forms a low-dimensional manifold that visualizes the classification decision patterns. Explanation on each individual sample can be then achieved in a counter-factual generation manner which continuously modifies the sample in one direction, by shifting its class-associated code along a guided path, until its classification outcome is changed. We compare our method with state-of-the-art ones on explaining image classification tasks in the form of saliency maps, demonstrating that our method achieves higher accuracies. The class-associated manifold not only helps with skipping local traps and achieving accurate explanation, but also provides insights to the data distribution patterns that potentially aids knowledge discovery.


## 0. Main Environments
```bash
conda create -n CAE python=3.9
conda activate CAE
pip install torch==1.12.1
pip install torchvision==0.13.1
pip install numpy==1.24.4
pip install pandas==1.3.5
pip install scikit-learn==1.0.2
pip install pillow==9.0.1
pip install opencv-python==4.10.0.84
pip install opencv-python-headless==4.10.0.84
pip install openpyxl==3.1.5
pip install networkx==3.1
pip install argparse==1.1
```



## 1. Prepare the dataset
## Original datasets could be downloaded from:
The retinal Optical Coherence Tomography (OCT) and the Chest X-rays image datasets are available at https://data.mendeley.com/datasets/rscbjbr9sj/2.
The Brain Tumor dataset 1 can be downloaded from https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection.
The Brain Tumor dataset 2 can be found at https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1.


## Data used for training and testing in our paper is available at:
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

## 2. Train the CAE
Open the 'CAE_Train' folder and run the 'main_train.py' file, then the CAE model starts training
```bash
cd code/CAE_Train
python main_train.py  # Train and test CAE model.
```

## 3. Obtain the trained CAE models and generated results of some cases
- After trianing, you could obtain the trained models in '/results/models/'.

  The trained CAE models that are used for extracting class-associated codes and generating new samples are named 'gen_index.pt' (where 'index' represents the iteration number of training).

- After trianing, you could obtain the generated results of some cases in '/results/images_display/'.

  In the '/results/images_display/' folder, for each image file, the first row is the original cases, the second row presents the samples generated by combining the class-associated codes and the individual codes from the original cases, the third row is the donor samples whose class-associated codes are extracted, and the fourth row presents the samples generated by combining the individual codes from the original samples in the first row with the class-associated codes from the samples in the third row.
  
  In the '/results/images_display/' folder, the file name, for example, 'gen_a2b_test2_00104000.jpg', means the generated results from A class (normal) to B class (abnormal) on test2 group samples using trained CAE model which has been trained 104000 iterations.


## 4. Perform class-associated codes analysis on test datasets
Open the 'CL_Analysis' folder
```bash
cd code/CL_Analysis
```

Run the 'CL_codes_extract.py' file, so we can extract class-associated codes of the test images using trained CAE model, which is put into the './code/trained_models/' folder.
The class-associated codes extracted from the test dataset are put into the './code/CL_Analysis/results/testAB_CL_codes_extraction_results.csv' file. Each code consists of 8 values.
```bash
python CL_codes_extract.py  # Extract class-associated codes of the test images using trained models.
```


Run the 'tsne_analysis.py' file, so we can perform t-SNE analysis on extracted class-associated codes.
The t-SNE analysis result is presented in the './code/CL_Analysis/results/tsne_analysis_result.png' file, where different numbers with different colors represent samples with different classes. 
```bash
python tsne_analysis.py  #  Perform t-SNE analysis on extracted class-associated codes.
```


## 5. Perform instance explanation using the class-associated manifold
- We can follow these steps for performing samples generation and instance explanation:

5.1 Open the 'code/Case_Show/' folder.
```bash
cd code/Case_Show
```

5.2 Run the 'local_explanation_on_instance.py' file to generate saliency map for instance explanation.
Along the path from the example to the reference counter example, we obtain meaningful class-associated codes for guided counterfactual generation, and by analyzing the changes of the generated samples and the changes of the outputs of the black-box model on the generated samples, we can get one saliency map for the instance explanation.
The generated samples are put into the './code/Case_Show/results/generate_img_saliency_maps_for_local_explanation/' folder. For example, 'ex_z_86_01567.png_ref_z_86_01318.png_d_0.2_gen.png' is the generated sample obtained by combining the individual code of the explained sample 'z_86_01567.png' and the class-associated code sampled at 0.2d distance from the example to the reference counter example, and here the 'ref_z_86_01318.png' indicates that the counter reference image is z_86_01318.png.
The saliency map obtained for instance explanation is also put into the './code/Case_Show/results/generate_img_saliency_maps_for_local_explanation/' folder.
```bash
python local_explanation_on_instance.py  #  Along the guided path (from example to the goal counter example), we obtain meaningful class-associated codes for guided counterfactual generation, and by analyzing the changes of the generated samples and the changes of the outputs of the black-box model on the generated samples, we can get one saliency map for the instance explanation. 
```


