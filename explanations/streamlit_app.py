import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import importlib
import os
from pathlib import Path
from omegaconf import OmegaConf
from dataset.classificationdataset import ImageClassificationDataset

# Load configuration
cfg = OmegaConf.load("params.yaml")

# Set title
st.title("Alzheimer's MRI Classification")

# Upload image
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    tensor
