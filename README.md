# Deep Learning-Based Classification of Alzheimer's Disease Using Augmented Brain MRI Scans

We build a PyTorch-based image classification model to detect Alzheimer’s disease stages (e.g., Normal, Mild Cognitive Impairment, Dementia) using MRI scans.

## Project Overview

We also:

    Use data augmentation to improve generalization.

    Track experiments and versions using DVC.

    Build a user interface using Streamlit for uploading MRI images and viewing predictions.

    Add explainability using Alibi or Captum to highlight the image regions influencing predictions.


## Recommended Dataset
Use a publicly available dataset like:

ADNI preprocessed MRI images (via Kaggle)
Contains 4 classes: MildDemented, ModerateDemented, NonDemented, VeryMildDemented.

## Directory Structure (Updated)

alzheimers_mri_classifier/<br>
├── data/<br>
│   └── raw/                   # Raw MRI scans<br>
├── models/<br>
│   └── model.pt               # Trained model<br>
├── src/<br>
│   ├── train.py               # PyTorch training script<br>
│   ├── dataset.py             # Data loading + transforms<br>
│   ├── explain.py             # Saliency map/IG explanation<br>
├── app/<br>
│   └── streamlit_app.py       # Streamlit web UI<br>
├── explanations/<br>
│   └── explanation_sample.png<br>
├── params.yaml                # Model hyperparameters<br>
├── dvc.yaml                   # DVC pipeline<br>
├── pyproject.toml             # Poetry setup<br>
└── README.md<br>


## Tools

| Tool        | Purpose                         |
| ----------- | ------------------------------- |
| PyTorch     | Model training                  |
| torchvision | Image transforms, augmentations |
| Captum      | Explainable AI for PyTorch      |
| Streamlit   | Interactive UI                  |
| DVC         | Data/model versioning           |
| Poetry      | Dependency management           |


## Installation

1- install poetry

2- make a new environment

3- install required packages

4- activate the enviroment

```bash
poetry shell
```


## To run a DVC pipeline:
```bash 
dvc repro
```

## Run the Streamlit App
After navigating to the explanations directory, sun the following command:

```bash 
streamlit run streamlit_webui.py
```