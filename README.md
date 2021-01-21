# Spectral Prediction
This repository contains message passing neural networks for spectral predictions as described in the paper [Message Passing Neural Networks for Infrared Spectral Predictions](). The `chemprop-IR` architecture is an extension of `chemprop` described in the paper [Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00237) and available in the [chemprop GiHub repository](https://github.com/chemprop/chemprop). 

The new spectral features are described here. Please see `README_chemprop` for details on base functionality.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
  * [Option 1: Conda](#option-1-conda)
  * [Option 2: Docker](#option-2-docker)
  * [(Optional) Installing `chemprop` as a Package](#optional-installing-chemprop-as-a-package)
  * [Notes](#notes)
- [Data](#data)
- [Training](#training)
  * [Spectral Metrics](#spectral-metrics)
  * [Output Activation](#output-activation)
  * [Spectral Normalization](#spectral-normalization)
  * [Ensemble Noise](#ensemble-noise)
  * [Train/Validation/Test Splits](#train-validation-test-splits)
- [Predicting](#predicting)
- [Results](#results)

## Requirements

While it is possible to run all of the code on a CPU-only machine, GPUs make training significantly faster. To run with GPUs, you will need:
 * cuda >= 8.0
 * cuDNN

## Installation

Installation of `chemprop-IR` is identical to `chemprop`.

### Option 1: Conda

The easiest way to install the `chemprop` dependencies is via conda. Here are the steps:

1. Install Miniconda from [https://conda.io/miniconda.html](https://conda.io/miniconda.html)
2. `cd /path/to/chemprop-IR`
3. `conda env create -f environment.yml`
4. `source activate chemprop_IR` (or `conda activate chemprop_IR` for newer versions of conda)
5. (Optional) `pip install git+https://github.com/bp-kelley/descriptastorus`

The optional `descriptastorus` package is only necessary if you plan to incorporate computed RDKit features into your model (see [Additional Features](#additional-features)). The addition of these features improves model performance on some datasets but is not necessary for the base model.

Note that on machines with GPUs, you may need to manually install a GPU-enabled version of PyTorch by following the instructions [here](https://pytorch.org/get-started/locally/).

### Option 2: Docker

Docker provides a nice way to isolate the `chemprop` code and environment. To install and run our code in a Docker container, follow these steps:

1. Install Docker from [https://docs.docker.com/install/](https://docs.docker.com/install/)
2. `cd /path/to/chemprop`
3. `docker build -t chemprop .`
4. `docker run -it chemprop:latest /bin/bash`

Note that you will need to run the latter command with nvidia-docker if you are on a GPU machine in order to be able to access the GPUs. 

### (Optional) Installing `chemprop` as a Package

If you would like to use functions or classes from `chemprop` in your own code, you can install `chemprop` as a pip package as follows:

1. `cd /path/to/chemprop-IR`
2. `pip install -e .`

Then you can use `import chemprop` or `from chemprop import ...` in your other code.

### Notes

**PyTorch GPU:** Although PyTorch is installed automatically along with `chemprop-IR`, you may need to install the GPU version manually. Instructions are available [here](https://pytorch.org/get-started/locally/).

**kyotocabinet**: If you get warning messages about `kyotocabinet` not being installed, it's safe to ignore them.
   
## Data

`chemprop-IR` supports spectral datasets, a special case of regression.

The data file must be be a **CSV file with a header row**. For `spectra`, the header row should indicate smiles followed by specified frequencies or intensities. The targets for a given molecule should be positive real numbers and may represent but are not limited to intensity, absorbance, transmittance, and reflectance. For example:
```
smiles,400,402,404,...,3996,3998,4000
CC(O)CC(C)C#COC#CC(C)CC(C)O,0.00126727,0.00127544,0.00127477,...,0.00101993,0.00104027,0.00098078
CCc1cc(C=NNC(=O)c2ccncc2)ccn1,0.000304076,0.000307313,0.000309041,...,0.000695609,0.000705222,0.000680961
...
```

Phases or solvents can be specified as features in a separate file. Using one-hot encodings with the above example,
```
gas,liquid,KBr,nujolmull,CCl4
1,0,0,0,0
0,0,1,0,0
...
```

Example datasets are available [here](https://www.dropbox.com/s/ceu1zi2rk0bxz9a/ir_models_data.tar.gz?dl=0). The linked directory contains several examples. See `ir_models_data/computed_model/computed_spectra.csv` for an example of how to store spectral data. See `ir_models_data/experiment_model/example_of_phase_feature_order.csv` for an example of how to store phase data. See `ir_models_data/solvation_example/solvation_spectra.csv` and `ir_models_data/solvation_example/solvation_phases.csv` for an example of how to store associated spectral and phase data. The latter describes data with artificial solvent shifts.

## Training

To train a model, run:
```
python train.py --data_path <path> (--features_path <features>) --dataset_type spectra --save_dir <dir>
```
where `<path>` is the path to a CSV file containing a spectral dataset, `<features>` is the path to a CSV file containing additional features such as phase or solvent information,`dataset_type` is set to `spectra`, and `<dir>` is the directory where model checkpoints will be saved.

For example:
```
python train.py --data_path ir_models_data/solvation_example/solvation_spectra.csv --features_path ir_models_data/solvation_example/solvation_phases.csv --dataset_type spectra --save_dir ir_checkpoints
```
We recommend using the training configuration in `ir_models_data/recommended_config.json` for the provided datasets. Configs can be specified by `--config_path <config>`. We also recommend using GPUs. This is controlled by the flag `--gpu <index>` where `<index>` specifies which GPU to use, if available. 

### Spectral Metrics

One novelty of our extensions to chemprop is the use of spectral metrics and loss functions. These metrics show improved performance over the application of standard metrics in multitask regression. The default metric and loss function for `spectra` is SID. Other metrics `<metric>` and losses `<loss>` may be specified with `--metric <metric>` and `--spectral_loss_function <loss>`.

Thresholds and regularizers are used to avoid singularities in evaluation of spectral metrics and loss functions. These are controlled by flags `--sm_thresh <thresh>` and `--sm_eps <eps>`. The threshold is set by `<thresh>`. All `spectra` values below `<thresh>` are set to `<thresh>`. The regularization is set by `<eps>`. This feature is enabled currently for the spectral RMSE.

### Output Activation

A final activation layer can be added at the end of the readout layer to enforce positive output values. The current supported types of activation include exponential and ReLU. This are controlled by the flag `--output_activation <act>`. Specify `<act>` as `exp` for exponential or `ReLU` for ReLU activation. Otherwise, no activation is applied to the final outputs.

### Spectral Normalization

The output of a `chemprop-IR` model can be normalized so that a particular spectrum range will sum to one. This would occur after output activation, if any. The start and end indices for normalization are specified by `--normalization_start <start>` and `--normalization_end <end>`. These indices correspond to the target indices within the dataset specified by `<path>`. For the above example, the target headers of `path` are `400,402,404,...,3996,3998,4000`. The target header `400` corresponds to index `0`. The target header `4000` corresponds to index `1801`. To normalize over the whole spectrum, specify `--normalization_start 0` and do not specify an end.

### Ensemble Noise

The ensemble variance can be computed by the flag `--ensemble_variance`. If `dataset_type` is `spectra`, the roundrobin SID is evaluated. To apply a convolution to spectra before roundrobin SID, specify `--ensemble_variance_conv <conv>` where `<conv>` is the number of target column spacings to use as the standard deviation. Otherwise, the variance across ensemble predictions is calculated.

### Train/Validation/Test Splits

Our code supports the base `chemprop` methods of splitting data into train, validation, and test sets. The newest method is a `random_with_repeated_smiles`. This utilizes a random split but ensures all entries with equivalent smiles are contained solely in one of the splits, intended for cases where a molecule is present in multiple phases.
 
## Predicting

To load a trained model and make predictions, run `predict.py` and specify:
* `--test_path <path>` Path to the data to predict on. Format this file as a `.csv` file with only a single column, with the header row with the entry `smiles` and every subsequent row entered with the SMILES you would like to predict.
* `--features_path <path>` Path to the data features to predict on. Feature columns must be in the same order as they were input during model training.
* A checkpoint by using either:
  * `--checkpoint_dir <dir>` Directory where the model checkpoint(s) are saved (i.e. `--save_dir` during training). This will walk the directory, load all `.pt` files it finds, and treat the models as an ensemble.
  * `--checkpoint_path <path>` Path to a model checkpoint file (`.pt` file).
* `--preds_path` Path where a CSV file containing the predictions will be saved.

For example:
```
python predict.py --test_path ir_models_data/solvation_example/solvation_spectra.csv --features_path ir_models_data/solvation_example/solvation_phases.csv --checkpoint_dir ir_checkpoints --preds_path ir_preds.csv
```

We recommend using GPUs. This is controlled by the flag `--gpu <index>` where `<index>` specifies which GPU to use, if available.

## Results

We compared `chemprop-IR` additions against standard quadratic losses and engineered fingerprints. The details are provided in our manuscript [Message Passing Neural Networks for Infrared Spectral Predictions]().

Model files are provided [here](https://www.dropbox.com/s/ceu1zi2rk0bxz9a/ir_models_data.tar.gz?dl=0).
