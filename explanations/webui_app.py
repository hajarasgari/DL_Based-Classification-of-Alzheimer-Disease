# streamlit_app.py
import streamlit as st
from PIL import Image
import torch

import torchvision.transforms as transforms
import sys
import os
# Add the models folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import dummy_model

from models.dummy_model import SimpleCNN
model = SimpleCNN()


st.title("Alzheimer's MRI Classification")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)

    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, 1).item()
        st.write(f"Prediction: {pred}")
