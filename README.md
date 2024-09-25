# RFML_Ed_Material

This git repo contains education and training material on standard methods for the test and evaluation (T&E) of classification machine learning models.  Jupyter notebooks and supporting 
python files are provided for this demonstration on a radio frequency machine learning use case.  

This repo also contains a prototype AI test harness.  The prototype has the ability to load pytorch models, load test data sets, and implement scoring metrics.

## Requirements

The Jupyter notebooks require standard python machine learning libraries such as [PyTorch](https://pytorch.org/).  The full list of requirements is provided in *environment.yml*.  

The *py_waspgen* package is also required.


## Installation and Quickstart

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the prototype AI test harness class from the root directory of this repository.

```bash
conda env create -f environment.yml
conda activate rfml_education_material
jupyter lab
```


## Jupyter Notebook Descriptions

1. *Binary_Classification_RFML_Tutorial.ipynb*:  Notebook detailing the training and evaluation of a binary classifier.
2. *Multiclass_Classification_RFML_Tutorial.ipynb*:  Notebook detailing the training and evaluation of a multiclass classifier.
3. *AI_Test_Harness_Demo.ipynb*: Notebook for evaluating a model using the test harness prototype.
