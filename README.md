# LigandScreen

This is repository "*Proximity Learning-Enabled Ligand Prediction for Ni-Catalyzed Atroposelective Suzuki-Miyaura Cross-Coupling: Leveraging Pd Catalysis Knowledge for Ni Discovery*"

# System requirements
In order to run Jupyter Notebook involved in this repository, several third-party python packages are required. The versions of these packages in our station are listed below. To reproduce the machine learning results, please install packages with same version as below.

```
rdkit==2024.03.5
numpy==1.24.4
pandas==2.2.2
scikit-learn==1.3.0
scipy==1.11.1
matplotlib==3.7.1
morfeus==0.7.2
xgboost==1.7.6
lightgbm==4.2.0
```
In addition to installing the aforementioned third-party libraries, you also need to execute the following code in current folder to install the Python package `asymopt` included with this project.
```
pip install .
```
All test were executed under Ubuntu 18.04.
# Demo & Instructions for use
Here we provide several notebooks in `code` folder to demonstrate how to generate virtual ligand library and perform ligand screening.