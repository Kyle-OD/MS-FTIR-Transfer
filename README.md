# Sustainable AI: Spectral Transformer

Kyle O'Donnell

## Introduction

This repository contains work started during my Summer 2024 internship through the [DoD SMART Scholar Program]()

This is a long-term project that will be continued each Summer over my graduate program, and be updated intermittently throughout the school year.

## Goal

The goal of this project is to work towards developing a state-of-the-art deep learning model capable of understanding electrochemical spectra from various modalities.  Summer 2024's efforts have been focused on determining what will be required, both in terms of data and computational resources, to build a model to satisfy the goal of this project.

## Environment

This project uses Anaconda for environment management and dependencies.  After installing Anaconda, create the environment using
```
conda env create -f environment.yml
```
This will create an environment named `sustainable_ai`.  Activate this environment with:
```
conda activate sustainable_ai
```

The models in this repository were trained on an NVIDIA 4060 and an NVIDIA 3080.  In the case of different GPUs or CPU only training, the CUDA and PyTorch versions may need to be adjusted.

## Data

While we plan to use multiple types of electrochemical spectra eventually, the initial work and testing has used Mass Spectrometry (MS) data, as it is the most widely available in open datasets.  

We sourced our data from the following sources:
- MassBank of North America (MoNA) \[[download]()\]  
- Pacific Northwest Lab (PNNL) FTIR data \[[download]()\]
- GNPS \[[download]()\]

## Model

Many recent advances in deep learning have come from large data in conjunction with the transformer architecture.  In order to tokenize raw spectral data, an approach similar to the Vision Transformer was implemented to convert the raw data into tokens for use with a transformer.

Initial testing used an encoder only transformer network with a feed-forward neural network for classification.  Further testing used paired encoder and decoder transformer networks for sequence to sequence prediction from spectra to SMILES

## Evaluation

To evaluate the prediction of SMILES strings, the following metrics were used:
- Binary Crossentropy
- Valid SMILES Percentage
- Tanimoto Similarity
- Dice Similarity
- Levenshtein Distance (Edit Distance)

(only Binary Crossentropy is used as a training criterion)

## Future Work

We have identified some potential paths to explore in future work on this project:
- Archetectural and Computational Improvements:
    - Utilize all available train/test data, rather than a small subset
    - Increase latent dimensionality, degree of multiheaded attention, and number of transformer layers in encoder and decoder
    - Include additional evaluation metrics, specifically chemistry-based metrics, in training criteria
- Utilize a trained MS-SMILES model and transfer learning to derive a FTIR-SMILES transformer using much more limited public FTIR data
- Explore a multimodal spectral approach to ingest spectra from a variety of electrochemical sensing technologies

## Usage

### Data Preparation

#### Mass Spectrometry

MassBank of 

To process the MoNA data, functions in `MoNA_reader.py` can be used.  The format of the MoNA dataset is a list of nested JSON objects, which can be too large to process on a workstation.  If this is the case, you can split the list from the command line using `sed`, and prepending/appending list qualifiers `[/]` where necessary.

To convert a MoNA JSON file to csv for easier processing, use the following code:
```
from MoNA_reader import process_json_file

process_json_file(**path_to_json.json**, **path_to_save.csv**)
```

#### Fourier Transform Infrared

The FTIR data sourced from PNNL is saved in the SPC format, which is commonly used for FTIR data.  To read this information, we have used the utilities in `readspc.py` written and provided by Dr. Charles Davidson.  Extracting the FTIR spectra can be done following this format:

```
from readspc.py import read

header, subheader, x, y = read(**path_to_spc.SPC**)
```

To generate plots for all PNNL FTIR files, see code in `plot_ftir.ipynb`

### Model Training

 

#### Transformer: Classifier

All code for testing the encoder only transformer classifier can be found in `./transformer/classifier.py`

For usage, see notebook `ms_transformer_exploration.ipynb`

#### Transformer: MS to SMILES

Code for generating

### Transfer Learning
Future task
### Multimodal Spectral Transformer
Future task