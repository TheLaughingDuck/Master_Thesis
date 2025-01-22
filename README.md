# Multimodal AI for brain tumor classification in MR images 

Code for my master thesis project on brain tumor classification using MR images.
One can consider adding a graphical description of the project as well.
<!--<img src="README_graphical_abstract.png" width="300px" align="right" />-->

<!--[Link to publication](https://doi.org/10.3384/9789180757805) | [Link to citation bib](#reference)-->


**(optional) Key highlights:**
- **Point 1**: This is repository will make you want to cite this work.
- **Point 2**:
- **Point 3**:

**(when a journal or thesis) Abstract**

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

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
List of relevant acknowledgements and/or references.