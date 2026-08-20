import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ------------------------------
# Imports from our modules
# ------------------------------
from data_loader import get_transforms
from autoencoder import Autoencoder
from explain import grad_cam

# ------------------------------
# Device
# ------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------
# Load models (cached)
# ------------------------------
@st.cache_resource
def load_models():
    from torchvision import models as tvmodels
    classifier = tvmodels.resnet50(pretrained=False)
    classifier.fc = torch.nn.Linear(classifier.fc.in_features, 3)
    classifier.load_state_dict(torch.load('models/classifier_resnet50.pth', map_location=DEVICE))
    classifier.to(DEVICE)
    classifier.eval()

    autoencoder = Autoencoder().to(DEVICE)
    autoencoder.load_state_dict(torch.load('models/autoencoder.pth', map_location=DEVICE))
    autoencoder.eval()

    return classifier, autoencoder

classifier, autoencoder = load_models()
transform = get_transforms()
class_names = ['Elliptical', 'Spiral', 'Merger/Other']

# ------------------------------
# UI
# ------------------------------
st.set_page_config(layout="wide")
st.title("🌌 Cosmic Anomaly Hunter")

# File upload or sample
uploaded = st.file_uploader("Upload an astronomical image", type=['jpg','png','jpeg'])

if uploaded is not None:
    image = Image.open(uploaded).convert('RGB')
else:
    # Use the first image from metadata as demo
    df = pd.read_csv('data/metadata.csv')
    sample_file = df.iloc[0]['filename']
    # Adjust path to your actual image location
    image_path = os.path.join('data/images_gz2/images', sample_file)
    if not os.path.exists(image_path):
        st.error(f"Sample image not found at {image_path}. Please check your image folder.")
        st.stop()
    image = Image.open(image_path).convert('RGB')

# Preprocess
img_tensor = transform(image).unsqueeze(0).to(DEVICE)

# Predict
with torch.no_grad():
    logits = classifier(img_tensor)
    probs = torch.softmax(logits, dim=1)
    conf, pred = torch.max(probs, dim=1)
    pred_class = pred.item()
    predicted_name = class_names[pred_class]

# Anomaly score (reconstruction error)
with torch.no_grad():
    recon = autoencoder(img_tensor)
    mse = torch.mean((recon - img_tensor)**2).item()
    anomaly_score = mse * 1000   # scaling factor for display

# Grad-CAM heatmap
cam = grad_cam(classifier, img_tensor, target_class=pred_class)

# ------------------------------
# DISPLAY – FIXED: use st.caption instead of caption= in st.pyplot
# ------------------------------
col1, col2 = st.columns(2)
with col1:
    st.image(image, caption="Original", use_column_width=True)

with col2:
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.imshow(cam, cmap='jet', alpha=0.5, extent=[0, image.width, image.height, 0])
    ax.axis('off')
    st.pyplot(fig)                 # <-- no caption here
    st.caption("Grad‑CAM Heatmap") # <-- caption added separately

# Show results
st.write(f"**Prediction:** {predicted_name} (confidence: {conf.item():.2%})")
st.write(f"**Anomaly Score:** {anomaly_score:.2f} (higher = more unusual)")






# ---------- DISPLAY ----------
st.write("### Debug Info")
st.write(f"Image type: {type(image)}")
st.write(f"Image size: {image.size}")
st.write(f"Image mode: {image.mode}")

# Show the original image first (outside columns, to test)
st.image(image, caption="Test: Original Image", use_container_width=True)

# Now try the side-by-side view
col1, col2 = st.columns(2)
with col1:
    st.image(image, caption="Original", use_container_width=True)
with col2:
    try:
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.imshow(cam, cmap='jet', alpha=0.5, extent=[0, image.width, image.height, 0])
        ax.axis('off')
        st.pyplot(fig)
        st.caption("Grad‑CAM Heatmap")
    except Exception as e:
        st.error(f"Error creating heatmap: {e}")

st.write(f"**Prediction:** {predicted_name} (confidence: {conf.item():.2%})")
st.write(f"**Anomaly Score:** {anomaly_score:.2f} (higher = more unusual)")