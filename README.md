# Multi-sequence 3D MRI fusion for paediatric brain tumour classification using deep learning

My Master's Thesis, ([available here](https://liu.diva-portal.org/smash/record.jsf?pid=diva2%3A1983250&dswid=6331)) on brain tumour classification using 3D MR images.
<!--<img src="README_graphical_abstract.png" width="300px" align="right" />-->

<!--[Link to publication](https://doi.org/10.3384/9789180757805) | [Link to citation bib](#reference)-->

**Key highlights:**
- **Three-dimensional**  MR Images of paediatric brains with either Glioma, Medulloblastoma or Ependymoma tumours were analysed.
- **The Transformer-based** BrainSegFounder model was utilized for feature extraction, and a custom neural network performed classification.
- **Two CNN architectures,** the ResNet (2+1)D and Resnet Mixed Convolution models were utilized for classifications.

**Abstract**

Paediatric brain tumours are a severe type of diseases with treatment plans varying by diagnosis, which is typically determined by radiological imaging. Recent deep learning projects perform diagnosis classification using a single 3D magnetic resonance imaging sequence type, or fusing 2D sequences with improved performance. This thesis aims to investigate the possibility of diagnosis classification as well as the related tumour location classification using 3D MRI deep learning models trained on either a single, or multiple distinct 3D MRI sequence types.

In this project, 243 pairs of T1W-GD and T2W MRI sequences of patients with paediatric brain tumours were analysed. Two classification tasks were considered: braintumour diagnosis classification (classes glioma, ependymoma and medulloblastoma) and tumour location classification (classes infratentorial and supratentorial). Three models were considered; a (vision transformer) Swin UNETR architecture trained by [1] named BrainSegFounder connected to a dense neural network, the ResNet (2+1)D and the ResNetMixed from [2] (both convolutional neural networks). The models utilized either early or intermediate fusion of sequences. The models were evaluated with various metrics: class-wise precision, class-wise recall, overall accuracy, balanced accuracy and balanced ROC-AUC. Models trained on single sequences and multiple sequences respectively were statistically compared using Wilcoxon signed rank tests on the predicted probability of the true class of each observation across models.

For diagnosis classification, overall classification performance was poor compared to random chance. The models were generally able to achieve high precision and recall values for glioma, and very low values for ependymoma and medulloblastoma. No model achieved balanced accuracy above 0.5, and the balanced ROC-AUC values were in the range [0.5, 0.66]. One model achieved significantly better true class prediction probabilities when trained on fused T1W-GD and T2W sequences as compared to T1W-GD sequences, but classifications were still poor by both models. The poor performance was investigated, and deemed to likely be caused by over-fitting as a consequence of the limited training data.

For the tumour location classification task, the models were able to achieve predictions marginally better than random chance. Recall values were larger than the class proportions, and balanced ROC-AUC values were in the range [0.7, 0.82]. No model trained on paired T1W-GD and T2W sequences was significantly better than corresponding single sequence models. The ResNet (2+1)D model achieved balanced accuracies 0.63, 0.59, 0.68 when trained on T1W-GD, T2W, and fused T1W-GD and T2W sequences respectively. The ResNet Mixed model achieved balanced accuracies 0.8, 0.68, 0.66 when trained on T1W-GD, T2W, and fused T1W-GD and T2W sequences respectively.

## Table of Contents
- [Prerequisites and setup](#Prerequisites-and-setup)
- [Datasets](#datasets)
- [Code Structure](#code-structure)
- [Usage](#usage)
- [(optional) Reference](#reference)
- [License](#license)

## Prerequisites and setup
A computer with GPU and relative drivers installed.
Anaconda or Miniconda Python environment manager.

### Setup
1. Clone the repo to your local machine
```sh
git clone git@github.com:something/something
```
2. Move to the downloaded repository and create a Python environment using the given .yml file
  ```sh
   conda env create -f environment_setup.yml
   ```

## Datasets
Short description of the dataset used in the project and link to it if open-source. 

## Code structure
Description of how the code is structured. One can consider the following structure as a starting point.
- **core**: folder containing the core functionality for data preprocessing, model training and testing, evaluation, plotting, etc. In here one can separate the scripts based on their function e.g. `core/preprocessing/generate_training_validation_test_splits.py` or `core/classification/run_model_classification.py`.
- (optional) **config**: consider using hydra configuration files to allow for easy track of the settings that were used for data preprocessing, model training and testing, etc. You can find more information on how to use them here [here](https://hydra.cc/docs/intro/) and can also take a look at [this repository](https://github.com/IulianEmilTampu/BTB_DEEP_LEARNING) where more advanced hydra configuration was used to set up project-wide configurations.
- **outputs**: folder where all the code outputs are saved. Consider using time stamps when saving your outputs. This folder can be part of your .gitignore file to avoid pushing ion the remote repository trained models and all the intermediate outputs. 

## Usage
Description of how the code should be used.

## Reference
If you use this work, please cite:

```bibtex
add bibtex reference here
```

## License
Consider which license should this work be covered by. Here is an example:
This work is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Acknowledgments
I would like to thank my supervisors, Anders Eklund and Neda Haj-Hosseini, for their
invaluable guidance throughout this project. I am also grateful to Iulian Emil Tampu and
Christoforos Spyretos for their many helpful contributions. Thank you also to Krzysztof
Bartoszek, for acting as examiner, and my opponent Duc Tran who suggested many useful
revisions.
