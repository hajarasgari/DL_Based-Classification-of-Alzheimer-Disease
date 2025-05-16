import streamlit as st
import sys
import types
# 🔧 Fix for torch.classes bug with Streamlit
sys.modules["torch.classes"] = types.ModuleType("torch.classes")
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
    tensor = transform(image).unsqueeze(0)

    # Load sample to infer shape & get classes
    val_dir = os.path.join(cfg.dataset.output_data_path, "val")
    val_dataset = ImageClassificationDataset(val_dir, transform=transform)
    class_names = val_dataset.get_class_names()

    # Import model class dynamically
    target = cfg.model._target_
    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)

    # Load model
    checkpoint_path = Path(os.getcwd()) / "best-checkpoint.ckpt"
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        in_channels=tensor.shape[1],
        num_classes=len(class_names)
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Inference
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).item()

    st.success(f"Prediction: **{class_names[pred]}**")
