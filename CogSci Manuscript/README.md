# Augmenting EEG with Generative Adversarial Networks Enhances Brain Decoding Across Classifiers and Sample Sizes
## Williams*, Weinhardt*, Wirzberger, & Musslick 
## 2023, Cognitive Science
### *Co-First Authors 

## Abstract

There is major potential for using electroencephalography (EEG) in brain decoding that has been untapped due to the need for large amounts of data. Advances in machine learning have mitigated this need through data augmentation techniques, such as Generative Adversarial Networks (GANs). Here, we gauged the extent to which GANs can augment EEG data to enhance classification performance. Our objectives were to determine which classifiers benefit from GANaugmented EEG and to estimate the impact of sample sizes on GAN-enhancements. We investigated three classifiers—neural networks, support vector machines, and logistic regressions— across seven sample sizes ranging from 5 to 100 participants. GAN-augmented EEG enhanced classification for neural networks and support vector machines, but not logistic regressions. Further, GAN-enhancements diminished as sample sizes increased—suggesting it is most effective with small samples, which may facilitate research that is unable to collect large amounts of data.

## Evaluation of the EEG-GAN Package

The [EEG-GAN](https://autoresearch.github.io/EEG-GAN/) package was developed in conjunction with this manuscript; however, this repository is the evaluation of this package, rather than the package itself. As can be seen within the manuscript, we evaluated whether EEG-GAN can produce realistic EEG data and augment this EEG data to improve classification performance. The data used in this manuscript was drawn from [Williams et al.'s, 2021](https://onlinelibrary.wiley.com/doi/abs/10.1111/psyp.13722) [open-source](https://osf.io/65x4v/) study.