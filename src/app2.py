import html
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from autoencoder import Autoencoder


st.set_page_config(
    page_title="Cosmic Anomaly Hunter",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

CLASS_NAMES = ["Elliptical", "Spiral", "Merger / Other"]
CLASS_DESCRIPTIONS = {
    "Elliptical": "A smooth, relatively featureless galaxy morphology with little visible spiral structure.",
    "Spiral": "A structured morphology characterised by a disk and visible spiral-arm features.",
    "Merger / Other": "An unusual, disturbed, irregular, merged or ambiguous galaxy morphology.",
}


def state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break
    return {key.removeprefix("module."): value for key, value in checkpoint.items()}


@st.cache_resource(show_spinner=False)
def load_classifier(weights_path):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(state_dict(torch.load(weights_path, map_location=DEVICE)))
    return model.to(DEVICE).eval()


@st.cache_resource(show_spinner=False)
def load_autoencoder(weights_path):
    model = Autoencoder()
    model.load_state_dict(state_dict(torch.load(weights_path, map_location=DEVICE)))
    return model.to(DEVICE).eval()


def grad_cam(model, tensor, class_index):
    """Returns a 224 x 224 Grad-CAM heatmap for the predicted ResNet50 class."""
    activations, gradients = [], []

    def save_activation(_, __, output):
        activations.append(output)

    def save_gradient(_, grad_input, grad_output):
        gradients.append(grad_output[0])

    layer = model.layer4[-1].conv3
    forward_hook = layer.register_forward_hook(save_activation)
    backward_hook = layer.register_full_backward_hook(save_gradient)
    try:
        model.zero_grad(set_to_none=True)
        model(tensor)[0, class_index].backward()
        weights = gradients[0][0].mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * activations[0][0]).sum(dim=0))
        cam = F.interpolate(
            cam[None, None], size=(224, 224), mode="bilinear", align_corners=False
        )[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy()
    finally:
        forward_hook.remove()
        backward_hook.remove()


def cam_overlay(image, heatmap):
    original = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255
    colour = np.zeros_like(original)
    colour[..., 0] = heatmap
    colour[..., 1] = 0.20 * (1 - heatmap)
    colour[..., 2] = 1 - heatmap
    return (255 * np.clip(0.60 * original + 0.40 * colour, 0, 1)).astype(np.uint8)


def get_anomaly_level(score):
    if score >= 2500:
        return "High"
    if score >= 1000:
        return "Moderate"
    return "Low"


st.title("🌌 Cosmic Anomaly Hunter")
st.caption(
    "AI-powered galaxy morphology classification, anomaly detection and visual explanation."
)
st.success("AI SYSTEM ONLINE")

with st.sidebar:
    st.header("Image selection")
    uploaded_file = st.file_uploader("Upload a galaxy image", type=["jpg", "jpeg", "png"])
    classifier_path = st.text_input("Classifier weights path", "/Users/ashwi/Downloads/MQAIS/Events/Cosmic Hunter/Models/classifier_resnet50.pth")
    autoencoder_path = st.text_input("Autoencoder weights path", "/Users/ashwi/Downloads/MQAIS/Events/Cosmic Hunter/Models/autoencoder.pth")
    st.caption(f"Runtime device: {DEVICE.type}")

if uploaded_file is None:
    st.info("Upload a galaxy image from the sidebar to begin analysis.")
    st.stop()

if not Path(classifier_path).is_file() or not Path(autoencoder_path).is_file():
    st.error("Model weights were not found. Correct both model paths in the sidebar.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
autoencoder_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

with st.spinner("Analysing the galaxy image…"):
    classifier = load_classifier(classifier_path)
    autoencoder = load_autoencoder(autoencoder_path)
    classifier_tensor = classifier_transform(image).unsqueeze(0).to(DEVICE)
    autoencoder_tensor = autoencoder_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probabilities = torch.softmax(classifier(classifier_tensor), dim=1)[0].cpu().numpy()
        reconstruction = autoencoder(autoencoder_tensor)
        raw_mse = F.mse_loss(reconstruction, autoencoder_tensor).item()

    predicted_index = int(probabilities.argmax())
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index] * 100)
    anomaly_score = raw_mse * 1000
    anomaly_level = get_anomaly_level(anomaly_score)
    overlay = cam_overlay(image, grad_cam(classifier, classifier_tensor, predicted_index))

st.header("📊 Analysis Overview")
st.caption("Neural-network analysis of the selected astronomical image.")
metrics = st.columns(4)
metrics[0].metric("Prediction", predicted_class, help="Most likely morphology class.")
metrics[1].metric("Confidence", f"{confidence:.1f}%", help="Classifier confidence.")
metrics[2].metric("Anomaly Score", f"{anomaly_score:.2f}", help="Model-derived reconstruction metric.")
metrics[3].metric("Anomaly Level", anomaly_level, help="Relative visual unusualness.")

probability_column, interpretation_column = st.columns(2, gap="large")
with probability_column:
    st.subheader("📈 Morphology Probability")
    for class_name, probability in zip(CLASS_NAMES, probabilities):
        percentage = float(probability * 100)
        st.write(f"**{class_name}** — {percentage:.1f}%")
        st.progress(int(round(percentage)))

with interpretation_column:
    st.subheader("🧠 AI Interpretation")
    st.write(
        f"The classifier strongly favours **{predicted_class}** morphology. "
        f"{CLASS_DESCRIPTIONS[predicted_class]} The model confidence is **{confidence:.1f}%**."
    )

image_column, cam_column = st.columns(2, gap="large")
with image_column:
    st.subheader("🔭 Selected Galaxy Image")
    st.image(image, use_container_width=True)
with cam_column:
    st.subheader("🔥 Grad-CAM Visual Explanation")
    st.image(overlay, use_container_width=True)
    st.caption("Highlighted regions contributed most strongly to the classification.")

anomaly_column, science_column = st.columns(2, gap="large")
with anomaly_column:
    st.subheader("🚨 Anomaly Assessment")
    st.metric("Reconstruction Error", f"{anomaly_score:.2f}")
    st.caption(f"Raw reconstruction MSE: {raw_mse:.6f}")
    st.write(
        f"**{anomaly_level} anomaly level.** The image has relatively "
        f"{anomaly_level.lower()} reconstruction error and may differ from patterns "
        "represented by the autoencoder."
    )

with science_column:
    st.subheader("🌠 Scientific Context")
    st.metric("Predicted Morphology", predicted_class)
    st.write(CLASS_DESCRIPTIONS[predicted_class])

st.header("🧭 Morphology Topology")
st.write(
    f"**{predicted_class}** — {CLASS_DESCRIPTIONS[predicted_class]} "
    "This is a model-derived visual classification, not a direct astronomical measurement."
)


st.header("⚙️ Model & Technical Details")
details = {
    "Galaxy Classifier": "ResNet50 — three-class morphology classification.",
    "Anomaly Detector": "Autoencoder — image reconstruction error as an anomaly indicator.",
    "Explainability": "Grad-CAM — regions associated with the classification.",
    "Output Classes": "Elliptical • Spiral • Merger / Other",
    "Runtime Device": DEVICE.type,
    "Image": html.escape(uploaded_file.name),
    "Anomaly Metric": "Model-derived reconstruction error; not a direct astronomical measurement.",
}
st.json(details)

st.divider()
st.caption(
    "Cosmic Anomaly Hunter • AI × Astronomy • Galaxy Morphology Analysis\n\n"
    "Predictions and anomaly scores are intended for exploratory analysis only."
)
