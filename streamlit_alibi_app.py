import streamlit as st
from PIL import Image
import types, sys
sys.modules["torch.classes"] = types.ModuleType("torch.classes")
import torch
import torchvision.transforms as transforms
import importlib
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from dataset.classificationdataset import ImageClassificationDataset
from alibi.explainers import KernelShap
import matplotlib.pyplot as plt
import alibi.explainers.shap_wrappers as shap_wrappers

import numpy as np

def visualize_image_grayscale(image, explanation, class_name=None):
    fig, ax = plt.subplots()
    ax.imshow(image, alpha=0.8)
    heatmap = ax.imshow(explanation, cmap='jet', alpha=0.5)
    fig.colorbar(heatmap, ax=ax)
    title = f"SHAP Explanation" + (f" for class '{class_name}'" if class_name else "")
    ax.set_title(title)
    ax.axis("off")
    return fig

def safe_rank_by_importance(shap_values, feature_names=None):
    importances = {}
    for class_idx, values in enumerate(shap_values):
        avg_mag_shap = np.mean(np.abs(values), axis=0)
        feature_order = np.argsort(avg_mag_shap)[::-1]
        most_important = avg_mag_shap[feature_order]

        most_important_names = []
        for i in feature_order:
            if isinstance(i, np.ndarray):
                if i.size == 1:
                    idx = int(i.item())
                else:
                    idx = int(i.flatten()[0])
            else:
                idx = int(i)
            most_important_names.append(feature_names[idx])

        importances[str(class_idx)] = {
            'ranked_effect': most_important,
            'names': most_important_names,
        }
    return importances
shap_wrappers.rank_by_importance = safe_rank_by_importance


# Load configuration
cfg = OmegaConf.load("params.yaml")

st.title("Alzheimer's MRI Classification with Explanation")

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
    val_dataset = ImageClassificationDataset(cfg.dataset.output_data_path + "/val", transform=transform)
    class_names = val_dataset.get_class_names()

    # Load model dynamically
    target = cfg.model._target_
    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    checkpoint_path = Path("best-checkpoint.ckpt")

    model = model_class.load_from_checkpoint(
        checkpoint_path,
        in_channels=tensor.shape[1],
        num_classes=len(class_names)
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tensor = tensor.to(device)

    # Inference
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).item()

    st.success(f"Prediction: **{class_names[pred]}**")

    # ========== Alibi Explain using Kernel SHAP ==========
    st.subheader("Explanation with SHAP")

    def predict_fn(x: np.ndarray):
        # Reshape from (N, 150528) back to (N, 224, 224, 3)
        x_reshaped = x.reshape((-1, 224, 224, 3))
        x_tensor = torch.tensor(x_reshaped, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            logits = model(x_tensor)
        return torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

    # Convert PIL image to numpy format for Alibi
    image_np = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    image_flat = image_np.reshape(1, -1)  # shape = (1, 224 * 224 * 3)
    baseline = np.zeros_like(image_flat)

    # Alibi SHAP Explainer
    explainer = KernelShap(predict_fn)
    # explainer.fit(np.array([np.zeros_like(image_np)]))  # baseline
    explainer.fit(baseline)
    explainer.feature_names = [f"pixel_{i}" for i in range(image_flat.shape[1])]  # avoid index error

    # explanation = explainer.explain(np.array([image_np]), nsamples=100)
    explanation = explainer.explain(image_flat, nsamples=100)

    # Reshape the output SHAP values back to image format for visualization:
    shap_vals_all = explanation.shap_values
    shap_vals_all = np.squeeze(shap_vals_all)  # → (150528, 4)
    shap_vals = shap_vals_all[:,pred]

    # Ensure it's a NumPy array before reshaping
    shap_vals = np.array(shap_vals)
    h = int(np.sqrt(shap_vals.size / 3))
    shap_vals = shap_vals.reshape(h, h, 3)

    shap_gray = np.mean(np.abs(shap_vals), axis=-1)

    # Visualize the explanation
    fig = visualize_image_grayscale(image_np, shap_gray, class_name=class_names[pred])

    st.pyplot(fig)
