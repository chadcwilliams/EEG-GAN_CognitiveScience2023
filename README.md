# Augmenting EEG with Generative Adversarial Networks Enhances Brain Decoding Across Classifiers and Sample Sizes
### Williams*, Weinhardt*, Wirzberger, & Musslick 
### 2023, Cognitive Science
### *Co-First Authors 

## Evaluation of the EEG-GAN Package

The [EEG-GAN](https://autoresearch.github.io/EEG-GAN/) package was developed in conjunction with this manuscript; however, this repository is the evaluation of this package, rather than the package itself. As can be seen within the manuscript, we evaluated whether EEG-GAN can produce realistic EEG data and augment this EEG data to improve classification performance. The data used in this manuscript was drawn from [Williams et al.'s, 2021](https://onlinelibrary.wiley.com/doi/abs/10.1111/psyp.13722) open-source study.

## Folder Breakdown

This study was a large undertaking and so the repository is quite dense. Here, we will break down each folder as a method of showing you our workflow. 

- Data: 
    - Plotting Datasets: This folder contains a few datafiles used for plotting
    - Training Datasets: This folder contains all 35 datasets used in GAN training and classification analyses.
    - Validation and Test Datasets: This folder contains the validation and test datasets that were used to determine performance on all classifications.