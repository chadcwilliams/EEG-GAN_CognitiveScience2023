# Augmenting EEG with Generative Adversarial Networks Enhances Brain Decoding Across Classifiers and Sample Sizes
Williams*, Weinhardt*, Wirzberger, & Musslick<br>
2023, Cognitive Science<br>
*Co-First Authors 

## Evaluation of the EEG-GAN Package

The [EEG-GAN](https://autoresearch.github.io/EEG-GAN/) package was developed in conjunction with this manuscript; however, this repository is the evaluation of this package, rather than the package itself. As can be seen within the manuscript, we evaluated whether EEG-GAN can produce realistic EEG data and augment this EEG data to improve classification performance. The data used in this manuscript was drawn from [Williams et al.'s, 2021](https://onlinelibrary.wiley.com/doi/abs/10.1111/psyp.13722) open-source study.

## Folder and File Breakdown

This study was a large undertaking and so the repository is quite dense. Here, we will break down each folder as a method of showing you our workflow. 

<b>Data</b>: 
- <b>Full Dataset</b>: This folder contains the pre-split datafile.
- <b>Training Datasets</b>: This folder contains all 35 datasets used in GAN training and classification analyses.
- <b>Validation and Test Datasets</b>: This folder contains the validation and test datasets that were used to determine performance on all classifications.
- ```gansEEG_ExtractSampleSizeData.py```: This file is used to split the full training dataset into the 35 different datasets within the <b>Training Datasets</b> folder.

<b>EEG Processing</b>:
- ```extractERP.m```: This is a Matlab file (sorry, it's the only one!) that deals with Williams et al.'s (2021) preprocessed data. Each of their participants were kept in a .mat file and this file opens each up, extracts the needed information and saves it into a single csv file.

<b>Evaluation</b>:
- ```gansEEG_Quantitative_Evaluation.py```: This file contains the code that ran the quantitative evaluations within the manuscript - specifically, the Train Synthetic, Test Real and the Train Real, Test Real analyses
- ```gansEEG_Qualitative_Evaluation.py```: This file contains the code that plots all qualitative evaluations.

<b>GANs</b>:
- <b>GAN Generated Data</b>: These are the GAN-generated artificial data created for each dataset.
- <b>GAN Models</b>: These are the trained GANs for each dataset.
- ```gansTrainingRunsArray.sh```: This is a batch script to automatically train each GAN on the respective datasets.

<b>Modelling</b>:
- <b>Classification Results</b>: This folder contains the classification performance outcome files.
- ```gansEEG_Classification.py```: This script runs the classifications (Neural Network, SVM, Logistic Regression), which was used to determine empirical and augmented performance across the seven sample sizes.
- ```gansEEG_ClassificationPlot.ipynb```: This script creates the classification plot

## Workflow

Next, we will discuss our workflow from extracting EEG data to achieving results. 

<b>Step 1: Download the data</b>
- First, we downloaded the data from Williams et al.'s (2021) [open-source repository](https://osf.io/65x4v/). We used the files within the <b>Open Data and Scripts/Open Data/Processed Data</b> folder. This folder contains ten zip files that we downloaded, unzipped, and merged into a single folder. 

<b>Step 2: Extract data from files</b>
- Now that we had all .mat files in one place, we used the ```extractERP.m``` file to extract, downsample, and concatenate trial-by-trial data for each participant. This results in a file ```ganTrialERP_len100.csv``` (now housed in the <b>Data/Full Dataset</b> folder). 

<b>Step 3: Split data into training, validation, and test sets </b>
- We now have all participant trial-by-trial data in a csv file, but we want to split the data into a training set and a non-training set (which will be again split into a validation and test sets in the next step). 
- ```gansEEG_ExtractSampleSizeData.py``` (in the <b>Data</b> folder) first removed 400 participants to later be split equally into validation and test sets and saved these as ```gansTrialERP_len100_TestValidationData.csv``` within the <b>Data/Validation and Test Datasets</b> folder. With the remaining 100 participants, it creates a series of new datasets with different sample sizes. It creates five sets of data for each sample size. These files can be found in the <b>Data/Training Datasets</b> folder. 

<b>Step 4: Split data into the test and validation datasets</b>
- In the last step, we created a file ```gansTrialERP_len100_TestValidationData.csv``` that contains all test and validation data. We next need to split these into two equally sized data. The file for this is absent from the repo but follows the same procedure as lines 13-21 in ```gansEEG_ExtractSampleSizeData.py```
- TODO: Find this file I guess (Note to self: is it just split in half? so the first half is one file and the second another)

<b>Step 5: Train the GANs</b>
- Next, we needed to train our GANs. We trained a single GAN on the full dataset with all participants ```gansTrialERP_len100.csv``` for our evaluations and then trained a GAN for each of our 35 training datasets. We did this by running the gan training file via terminal. As we were using a super computer for this, we set up batch job, specifically the ```gansTrainingRunsArray.sh``` file in the <b>GANs</b> folder. In the last line of this file you can see the training parameters we used, and the batch job simply iterates through all of our training files one at a time. 
- <i>Note that the EEG-GAN package requires a specific folder structure - it requires <b>data</b>, <b>trained_models</b>, and </b>generated_samples</b> folders. So, this step required that we moved our data being trained into a folder named data. This folder does not exist in the current repo, but would be an extra step required for replication of our study</i>

<b>Step 6: Generate samples</b>
- We then generated samples for each of the GANs trained in the previous step. We did this using terminal commands manually, so have no batch file this time. The generated samples of the GAN that was trained for evaluation (thus on all data) can be found as ```sd_len100_30000ep.csv``` within the <b>GANs/GAN Generated Data</b> folder. The remainder of the files in this folder are the samples generated for each of the training datasets.

<b>Step 7: Evaluation</b>
- We ensured the GAN can learn to generate realistic EEG data using both qualitative and quantitative anaylses. For qualitative analyses, we created a series of plots comparing empirical versus synthetic data via the ```gansEEG_Qualitative_Evaluation.ipynb``` file within <b>Evaluations</b> folder. For quantitative analyses, we followed the Train Synthetic, Test Real approach as can be seen in the ```gansEEG_Quantitative_Evaluation.py``` file within the <b>Evaluations</b> folder.

<b>Step 8: Classification</b>
- The main findings of our study are the classifications of both empirical and augmented data within neural network, suppot vector machine, and logistic regression classifiers. These analyses were conducted via the ```gansEEG_Classification.py``` file within the <b>Modelling</b> folder. This script outputted classification results, which are stored within the <b>Modelling/Classification Results</b> folder. 

<b>Step 9: Plot the classification</b>
- The final step was to plot our classification results. The figures within the manuscript mostly stem from the evaluation procedure in Step 7, except for this final figure which is created via the ```gansEEG_ClassificationPlot.ipynb``` file within the <b>Modelling</b> folder.
