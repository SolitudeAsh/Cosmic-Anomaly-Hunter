import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os

from data_loader import get_transforms
from autoencoder import Autoencoder
from explain import grad_cam


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    from torchvision import models as tvmodels

    # -------------------------
    # Classifier
    # -------------------------

    classifier = tvmodels.resnet50(weights=None)

    classifier.fc = torch.nn.Linear(
        classifier.fc.in_features,
        3
    )

    classifier.load_state_dict(
        torch.load(
            "models/classifier_resnet50.pth",
            map_location=DEVICE
        )
    )

    classifier.to(DEVICE)
    classifier.eval()


    # -------------------------
    # Autoencoder
    # -------------------------

    autoencoder = Autoencoder().to(DEVICE)

    autoencoder.load_state_dict(
        torch.load(
            "models/autoencoder.pth",
            map_location=DEVICE
        )
    )

    autoencoder.eval()


    return classifier, autoencoder


classifier, autoencoder = load_models()

transform = get_transforms()

class_names = [
    "Elliptical",
    "Spiral",
    "Merger/Other"
]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Cosmic Anomaly Hunter",
    page_icon="🌌",
    layout="wide"
)


st.title("🌌 Cosmic Anomaly Hunter")

st.write(
    "Upload a galaxy image to classify its morphology "
    "and estimate how unusual it is."
)


# ============================================================
# IMAGE INPUT
# ============================================================

uploaded = st.file_uploader(
    "Upload an astronomical image",
    type=["jpg", "jpeg", "png"]
)


if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

else:

    # -------------------------
    # Demo image
    # -------------------------

    metadata_path = "Data/metadata.csv"

    if not os.path.exists(metadata_path):

        # Try lowercase data directory
        metadata_path = "data/metadata.csv"


    if not os.path.exists(metadata_path):

        st.error(
            "metadata.csv was not found. "
            "Please check your Data/ or data/ folder."
        )

        st.stop()


    df = pd.read_csv(metadata_path)


    # Check that filename exists

    if "filename" not in df.columns:

        st.error(
            "metadata.csv does not contain a 'filename' column."
        )

        st.write(
            "Columns found:",
            df.columns.tolist()
        )

        st.stop()


    sample_file = df.iloc[0]["filename"]


    # -------------------------
    # Find sample image
    # -------------------------

    possible_paths = [

        os.path.join(
            "Data",
            "images_gz2",
            "images",
            sample_file
        ),

        os.path.join(
            "data",
            "images_gz2",
            "images",
            sample_file
        )

    ]


    image_path = None

    for path in possible_paths:

        if os.path.exists(path):

            image_path = path
            break


    if image_path is None:

        st.error(
            f"Sample image '{sample_file}' "
            "could not be found."
        )

        st.write("Checked:")

        for path in possible_paths:
            st.write(path)

        st.stop()


    image = Image.open(
        image_path
    ).convert("RGB")


# ============================================================
# PREPROCESS
# ============================================================

img_tensor = transform(
    image
).unsqueeze(0).to(DEVICE)


# ============================================================
# CLASSIFICATION
# ============================================================

with torch.no_grad():

    logits = classifier(
        img_tensor
    )

    probs = torch.softmax(
        logits,
        dim=1
    )

    conf, pred = torch.max(
        probs,
        dim=1
    )


pred_class = pred.item()

predicted_name = class_names[
    pred_class
]


# ============================================================
# ANOMALY SCORE
# ============================================================

with torch.no_grad():

    recon = autoencoder(
        img_tensor
    )

    mse = torch.mean(
        (recon - img_tensor) ** 2
    ).item()


anomaly_score = mse * 1000


# ============================================================
# GRAD-CAM
# ============================================================

cam = grad_cam(
    classifier,
    img_tensor,
    target_class=pred_class
)


# ============================================================
# DISPLAY IMAGES
# ============================================================

col1, col2 = st.columns(2)


# -------------------------
# Original
# -------------------------

with col1:

    st.image(
        image,
        caption="Original Galaxy",
        use_container_width=True
    )


# -------------------------
# Grad-CAM
# -------------------------

with col2:

    fig, ax = plt.subplots()

    ax.imshow(image)

    ax.imshow(
        cam,
        cmap="jet",
        alpha=0.5,
        extent=[
            0,
            image.width,
            image.height,
            0
        ]
    )

    ax.axis("off")

    st.pyplot(
        fig,
        use_container_width=True
    )

    st.caption(
        "Grad-CAM: regions that influenced the classifier"
    )

    plt.close(fig)


# ============================================================
# RESULTS
# ============================================================

st.divider()

st.subheader("🔭 Classification Result")

result_col1, result_col2 = st.columns(2)


with result_col1:

    st.metric(
        "Predicted Galaxy Type",
        predicted_name
    )


with result_col2:

    st.metric(
        "Classification Confidence",
        f"{conf.item():.2%}"
    )


st.subheader("🚨 Anomaly Detection")

st.metric(
    "Anomaly Score",
    f"{anomaly_score:.2f}"
)

st.caption(
    "Higher scores indicate greater reconstruction error "
    "and therefore potentially more unusual images."
)


# ============================================================
# PROBABILITIES
# ============================================================

st.subheader("Class Probabilities")

probability_df = pd.DataFrame(
    {
        "Galaxy Type": class_names,
        "Probability": probs[0].detach().cpu().numpy()
    }
)

st.bar_chart(
    probability_df.set_index("Galaxy Type")
)


# ============================================================
# IMAGE INFORMATION
# ============================================================

with st.expander("Image Information"):

    st.write(
        f"**Image size:** {image.size[0]} × {image.size[1]}"
    )

    st.write(
        f"**Image mode:** {image.mode}"
    )

    st.write(
        f"**Device:** {DEVICE}"
    )