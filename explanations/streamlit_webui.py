import sys
import types

import streamlit as st
from PIL import Image

sys.modules["torch.classes"] = types.ModuleType("torch.classes")
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf

sys.path.append(os.path.abspath(".."))
import alibi.explainers.shap_wrappers as shap_wrappers
import matplotlib.pyplot as plt
import numpy as np
from alibi.explainers import KernelShap
from captum.attr import IntegratedGradients
from lime import lime_image
from skimage.segmentation import mark_boundaries
from sklearn.metrics.pairwise import cosine_similarity

from dataset.classificationdataset import ImageClassificationDataset


class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def visualize_image_grayscale(image, explanation, class_name=None):
    fig, ax = plt.subplots()
    ax.imshow(image, alpha=0.8)
    heatmap = ax.imshow(explanation, cmap="jet", alpha=0.5)
    fig.colorbar(heatmap, ax=ax)
    title = f"SHAP Explanation" + (f" for class '{class_name}'" if class_name else "")
    ax.set_title(title)
    ax.axis("off")
    return fig


def visualize_integrated_gradients(image, attributions, class_name=None):
    attributions = attributions.squeeze().cpu().numpy().transpose(1, 2, 0)
    attributions = np.mean(np.abs(attributions), axis=-1)
    fig, ax = plt.subplots()
    ax.imshow(image, alpha=0.8)
    heatmap = ax.imshow(attributions, cmap="viridis", alpha=0.5)
    fig.colorbar(heatmap, ax=ax)
    title = f"Integrated Gradients" + (f" for class '{class_name}'" if class_name else "")
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
            "ranked_effect": most_important,
            "names": most_important_names,
        }
    return importances


shap_wrappers.rank_by_importance = safe_rank_by_importance

# Load configuration
cfg = OmegaConf.load("../params.yaml")

st.title("Alzheimer's MRI Classification with Explanation")

# Upload image
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Define transform
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    tensor = transform(image).unsqueeze(0)

    # Load sample to infer shape & get classes
    # val_dataset = ImageClassificationDataset(cfg.dataset.output_data_path + "/val", transform=transform)
    val_dataset = ImageClassificationDataset("../data/augmented_alzheimer/val", transform=transform)
    class_names = val_dataset.get_class_names()

    # Load model dynamically
    target = cfg.model._target_
    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    checkpoint_path = Path("../best-checkpoint.ckpt")

    model = model_class.load_from_checkpoint(checkpoint_path, in_channels=tensor.shape[1], num_classes=len(class_names))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tensor = tensor.to(device)

    # Add embedding extractor method
    if not hasattr(model, "get_embedding"):

        def get_embedding(self, x):
            with torch.no_grad():
                x = self.features(x)
                x = torch.flatten(x, 1)
            return x

        import types

        model.get_embedding = types.MethodType(get_embedding, model)

    # Inference
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).item()

    st.success(f"Prediction: **{class_names[pred]}**")

    # ========== Alibi Explain using Kernel SHAP ==========
    st.subheader("Explanation with SHAP")

    def predict_fn(x: np.ndarray):
        x_reshaped = x.reshape((-1, 224, 224, 3))
        x_tensor = torch.tensor(x_reshaped, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            logits = model(x_tensor)
        return torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

    image_np = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    image_flat = image_np.reshape(1, -1)
    baseline = np.zeros_like(image_flat)

    explainer = KernelShap(predict_fn)
    explainer.fit(baseline)
    explainer.feature_names = [f"pixel_{i}" for i in range(image_flat.shape[1])]

    explanation = explainer.explain(image_flat, nsamples=100)

    shap_vals_all = explanation.shap_values
    shap_vals_all = np.squeeze(shap_vals_all)
    shap_vals = shap_vals_all[:, pred]

    shap_vals = np.array(shap_vals)
    h = int(np.sqrt(shap_vals.size / 3))
    shap_vals = shap_vals.reshape(h, h, 3)

    shap_gray = np.mean(np.abs(shap_vals), axis=-1)

    fig = visualize_image_grayscale(image_np, shap_gray, class_name=class_names[pred])
    st.pyplot(fig)

    # ========== Integrated Gradients ==========
    st.subheader("Explanation with Integrated Gradients")
    wrapped_model = WrappedModel(model)
    ig = IntegratedGradients(wrapped_model)
    baseline = torch.zeros_like(tensor).to(device)
    attributions, _ = ig.attribute(tensor, baseline, target=pred, return_convergence_delta=True)
    fig_ig = visualize_integrated_gradients(image_np, attributions, class_name=class_names[pred])
    st.pyplot(fig_ig)

    # ========== LIME Explanation ==========
    # st.subheader("Explanation with LIME")

    # def lime_predict(images):
    #     batch = torch.stack([transform(Image.fromarray(img)).to(device) for img in images], dim=0)
    #     with torch.no_grad():
    #         outputs = model(batch)
    #     return torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

    # explainer_lime = lime_image.LimeImageExplainer()
    # explanation_lime = explainer_lime.explain_instance(
    #     np.array(image.resize((224, 224))),
    #     lime_predict,
    #     top_labels=1,
    #     hide_color=0,
    #     num_samples=1000
    # )

    # temp, mask = explanation_lime.get_image_and_mask(
    #     label=pred,
    #     positive_only=False,
    #     num_features=10,
    #     hide_rest=False
    # )

    # fig_lime, ax_lime = plt.subplots()
    # ax_lime.imshow(mark_boundaries(temp / 255.0, mask))
    # ax_lime.set_title(f"LIME Explanation for class '{class_names[pred]}'")
    # ax_lime.axis("off")
    # st.pyplot(fig_lime)

    # # ========== Similarity Explanation ==========
    # st.subheader("Most Similar MRI Images (Based on Embedding Similarity)")

    # reference_gallery = []
    # for img, label in val_dataset:
    #     img_tensor = img.unsqueeze(0).to(device)
    #     embedding = model.get_embedding(img_tensor).cpu().numpy()
    #     reference_gallery.append((embedding, img, label))

    # input_embedding = model.get_embedding(tensor).cpu().numpy().reshape(1, -1)

    # similarities = []
    # for emb, img_ref, label_ref in reference_gallery:
    #     sim = cosine_similarity(input_embedding, emb.reshape(1, -1))[0, 0]
    #     similarities.append((sim, img_ref, label_ref))

    # similarities.sort(reverse=True)
    # top_k = similarities[:5]

    # for score, img_sim, lbl in top_k:
    #     st.image(transforms.ToPILImage()(img_sim), caption=f"Class: {class_names[lbl]} — Similarity: {score:.2f}", use_column_width=True)
