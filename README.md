# CellDeathPred: a deep learning framework for ferroptosis and apoptosis prediction based on cell painting

Application of contrastive learning for ferroptosis and apoptosis indentification, coded up in Python by Aidin Biibosunov and Alaa Bessadok. Please contact biibosunov.aidin@gmail.com for further inquiries. Thanks.

<img width="641" alt="Screenshot 2023-09-25 at 14 05 57" src="https://github.com/peng-lab/CellDeathPred/assets/67750721/db12c435-3ca6-4c62-865d-e78026da2bea">

> **Abstract:** *Cell death, such as apoptosis and ferroptosis, play essential roles in the process of development, homeostasis, and pathogenesis of acute and chronic diseases. The increasing number of studies investigating cell death types in various diseases, particularly cancer
and degenerative diseases, has raised hopes for their modulation in disease therapies. However, identifying the presence of a particular cell death type is not an obvious task, as it requires computationally intensive work and costly experimental assays. To
address this challenge, we present CellDeathPred, a novel deep-learning framework that uses high-content imaging based on cell painting to distinguish cells undergoing ferroptosis or apoptosis from healthy cells. In particular, we incorporate a deep neural
network that effectively embeds microscopic images into a representative and discriminative latent space, classifies the learned embedding into cell death modalities, and optimizes the whole learning using the supervised contrastive loss function. We
assessed the efficacy of the proposed framework using cell painting microscopy data sets from human HT-1080 cells, where multiple inducers of ferroptosis and apoptosis were used to trigger cell death. Our model confidently separates ferroptotic and
apoptotic cells from healthy controls, with an average accuracy of 95% on non-confocal data sets, supporting the capacity of the CellDeathPred framework for cell death discovery*

This work is published in nature cell death discovery journal [[`nature CDD`](https://www.nature.com/articles/s41420-023-01559-y)] [[`bioRxiv`](https://www.biorxiv.org/content/10.1101/2023.03.14.532633v1)]

## Requirements

This codebase has been developed on a linux machine with python version 3.8, torch 1.8.1, torchvision 0.9.1 and a HPC cluster running with the slurm workload manager. All required python packages and corresponding version for this setup can be found in the [requirements.txt](requirements.txt) file.

## Pretrained models

The model that was used in the paper is [here](./Code/saved_models/saved_models_train_exp67_2)

## CellDeathPred training and evaluation on cell painting images

You can train a new model with the following command:

```bash
sh run.sbatch
```
TODO: add documentation for this command

If you want to test the code using the pretrained model, please use this [notebook](./Code/notebooks/dataset_exp3.ipynb)

## Citation

If our code is useful for your work please cite our paper:

```latex
Schorpp, K., Bessadok, A., Biibosunov, A. et al. CellDeathPred: a deep learning framework for ferroptosis and apoptosis prediction based on cell painting.
Cell Death Discov. 9, 277 (2023). https://doi.org/10.1038/s41420-023-01559-y
```
