# Augmenting EEG with Generative Adversarial Networks Enhances Brain Decoding Across Classifiers and Sample Sizes
### Williams*, Weinhardt*, Wirzberger, & Musslick 
### 2023, Cognitive Science
### *Co-First Authors 

## Evaluation of the EEG-GAN Package

The [EEG-GAN](https://autoresearch.github.io/EEG-GAN/) package was developed in conjunction with this manuscript; however, this repository is the evaluation of this package, rather than the package itself. As can be seen within the manuscript, we evaluated whether EEG-GAN can produce realistic EEG data and augment this EEG data to improve classification performance. The data used in this manuscript was drawn from [Williams et al.'s, 2021](https://onlinelibrary.wiley.com/doi/abs/10.1111/psyp.13722) open-source study.

## Folder and File Breakdown

This study was a large undertaking and so the repository is quite dense. Here, we will break down each folder as a method of showing you our workflow. 

<b>Data</b>: 
- <b>Plotting Datasets</b>: This folder contains a few datafiles used for plotting.
- <b>Training Datasets</b>: This folder contains all 35 datasets used in GAN training and classification analyses.
- <b>Validation and Test Datasets</b>: This folder contains the validation and test datasets that were used to determine performance on all classifications.
- ```gansEEG_ExtractSampleSizeData.py```: This file is used to split the full training dataset into the 35 different datasets within the <b>Training Datasets</b> folder.

<b>EEG Processing</b>:
- ```extractERP.m```: This is a Matlab file (sorry, it's the only one!) that deals with Williams et al.'s (2021) preprocessed data. Each of their participants were kept in a .mat file and this file opens each up, extracts the needed information and saves it into a single csv file.

<b>Evaluation</b>:
- ```gansEEG_NeuralNetwork_Evaluation.py```: This file contains the code that ran the quantitative evaluations within the manuscript - specifically, the Train Synthetic, Test Real and the Train Real, Test Real analyses

<b>GANs</b>:
- <b>GAN Generated Data</b>: These are the GAN-generated artificial data created for each dataset.
- <b>GAN Models</b>: These are the trained GANs for each dataset.
- ```gansTrainingRunsArray.sh```: This is a batch script to automatically train each GAN on the respective datasets.

<b>Modelling</b>:
- <b>Classification Results</b>: This folder contains the classification performance outcome files.
- <b>Models</b>: This folder contains scripts for each classifier (Neural Network, SVM, Logistic Regression), which was used to determine empirical and augmented performance across the seven sample sizes.

<b>Plotting</b>:
- <b>Figures</b>: This folder contains the manuscript figures.
- <b>Script</b>: This folder contains the scripts used to create the manuscript figures.

## Workflow

Next, we will discuss our workflow from extracting EEG data to achieving results. 

<b>Step 1: Download the data</b>
- Download the data from Williams et al.'s (2021) [open-source repository](https://osf.io/65x4v/). We used the files within <b>Open Data and Scripts</b> -> <b>Open Data</b> -> <b>Processed Data</b> folder. This folder contains ten zip files that we downloaded, unzipped, and merged into a single folder. 

Step 2: Extract data from files</b>