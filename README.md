# eae-transfer

Official repository for the paper "Small Models Are (Still) Effective Cross-Domain Argument Extractors" by William Gantt and Aaron Steven White.

## Getting Started

This repository uses [poetry](https://python-poetry.org/) for dependency management. If you don't already have poetry installed, you can follow the instructions for doing so [here](https://python-poetry.org/docs/#installation). Note that the poetry documentation strongly recommends installing the package inside a dedicated virtual environment. We used Conda for this purpose and have not tested using other virtual environment platforms.

This project has [PyTorch](https://pytorch.org/) as one of its dependencies. The version specified in the `pyproject.toml` file is specific to the ISA and the CUDA version we used for development (11.7). You will have to change that line to the version appropriate for your machine. We recommend still using PyTorch version 2.0.1 if possible. You can find the appropriate wheel for your setup [here](https://download.pytorch.org/whl/torch/).

With poetry installed and your virtual environment active, you can next run `poetry install` from the project root to install all necessary dependencies.

After this, you will have to install the `en_core_web_sm` pipeline for SpaCy:

```
python -m spacy download en_core_web_sm
```

Next, be sure to change the `ROOT` variable on L6 of `eae/dataset/common.py` to point to the project root.

Finally, unzip `data.zip`. You're now ready to go!

## Data

The experiments in our paper cover the following EAE datasets:

- ACE 2005 ([Doddington et al., 2004](https://aclanthology.org/L04-1011/))
- ERE-Light and ERE-Rich ([Song et al., 2015](https://aclanthology.org/W15-0812/))
- FAMuS ([Vashishtha et al., 2015](https://arxiv.org/abs/2311.05601))
- RAMS ([Ebner et al., 2020](https://aclanthology.org/2020.acl-main.718/))
- WikiEvents ([Li et al., 2021](https://aclanthology.org/2021.naacl-main.69/))

We have included FAMuS, RAMS, and WikiEvents in the `data.zip` archive in this repo, and our experiments with these datasets can be reproduced following the instructions below. We do not have permission to distribute the other three datasets, but they can be downloaded through the LDC (ACE 2005: LDC2006T06; ERE-Light and -Rich: LDC2023T0). If you are interested in reproducing results with these datasets and you have access to them, please create an issue and we will help you do so.

Preprocessing code can be found in `dataset/dataset.py` and in `preprocessing/preprocess_{qa,infilling}.py`. This code will be run automatically when you start training.

## Training

Scripts demonstrating how to invoke training for the Flan-T5 question answering (QA) and template infilling (TI) models are included in `scripts/training/{qa,ti}`. You may need to adapt these to your particular compute environment, but the 

## Inference

Scripts for inference are in `scripts/inference/{qa,ti}`. As with training, you will need to set certain environment variables here; see the scripts for details.

## Evaluation

Both training and evaluation scripts will print out exact match argument precision, recall, and F1 for all datasets. They will also print trigger F1, but this will always be 1.0, as trigger extraction is not part of the task.

## Questions

Please create an issue in this repository if you have any questions.