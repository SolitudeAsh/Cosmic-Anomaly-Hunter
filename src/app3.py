import base64
import html
import io
import random
import textwrap
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torchvision import models, transforms

from autoencoder import Autoencoder

st.set_page_config(
    page_title="Cosmic Anomaly Hunter",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SIDEBAR CONTROLS & SPECTRAL THEME SELECTION
# ============================================================
with st.sidebar:
    st.markdown("### 🔭 OBSERVATORY CONTROLS")
    st.caption("Configure the array before acquiring a target.")

    uploaded_file = st.file_uploader("Upload Deep-Space Image", type=["jpg", "jpeg", "png"])

    st.markdown("---")
    st.markdown("##### 🎨 HUD Spectral Theme")
    theme_choice = st.selectbox(
        "Select Spectrum Palette",
        ["Cyber Neon Cyan", "Deep Infrared", "Ultraviolet Blue", "Solar Gold"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("##### 🎛️ Telescope Optics")
    exposure_val = st.slider("Exposure / Brightness", 0.5, 2.0, 1.0, 0.1)
    contrast_val = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
    reticle_opacity = st.slider("Crosshair Reticle Opacity", 0.0, 1.0, 0.6, 0.1)
    heatmap_alpha = st.slider("Grad-CAM Overlay Blend", 0.0, 1.0, 0.4, 0.05)
    show_reticle = st.checkbox("Enable HUD Reticle Overlay", value=True)
    show_starfield = st.checkbox("Enable Starfield Backdrop", value=True)

    with st.expander("📡 Mission Briefing — what do these controls do?"):
        st.write(
            "- **Exposure / Contrast** adjust the raw optical feed before inference.\n"
            "- **Reticle Opacity** controls the targeting crosshair overlaid on every lens.\n"
            "- **Grad-CAM Blend** controls how strongly the thermal attention map is "
            "painted over the direct image.\n"
            "- Toggle the **starfield backdrop** off on slower machines for a calmer view."
        )

    with st.expander("⚙️ Model Weights Path"):
        classifier_path = st.text_input(
            "Classifier Path",
            "/Users/ashwi/Downloads/MQAIS/Events/Cosmic Hunter/Models/classifier_resnet50.pth",
        )
        autoencoder_path = st.text_input(
            "Autoencoder Path",
            "/Users/ashwi/Downloads/MQAIS/Events/Cosmic Hunter/Models/autoencoder.pth",
        )

    st.markdown("---")
    st.caption(f"🟢 SYSTEMS NOMINAL · {datetime.now().strftime('%H:%M:%S UTC')}")

THEMES = {
    "Cyber Neon Cyan": {
        "accent": "#00F0FF", "accent2": "#7CFFEA", "glow": "rgba(0, 240, 255, 0.4)",
        "bg": "#060810", "nebula1": "rgba(0, 240, 255, 0.16)", "nebula2": "rgba(120, 0, 255, 0.12)",
    },
    "Deep Infrared": {
        "accent": "#FF3366", "accent2": "#FF9E7A", "glow": "rgba(255, 51, 102, 0.4)",
        "bg": "#0C0608", "nebula1": "rgba(255, 51, 102, 0.16)", "nebula2": "rgba(255, 140, 0, 0.10)",
    },
    "Ultraviolet Blue": {
        "accent": "#9D00FF", "accent2": "#C084FF", "glow": "rgba(157, 0, 255, 0.4)",
        "bg": "#08060C", "nebula1": "rgba(157, 0, 255, 0.16)", "nebula2": "rgba(0, 140, 255, 0.10)",
    },
    "Solar Gold": {
        "accent": "#FFB020", "accent2": "#FFE08A", "glow": "rgba(255, 176, 32, 0.4)",
        "bg": "#0C0A06", "nebula1": "rgba(255, 176, 32, 0.16)", "nebula2": "rgba(255, 60, 60, 0.10)",
    },
}
active_theme = THEMES[theme_choice]


def _glow(alpha):
    """Return the theme glow colour with a custom alpha."""
    r, g, b = active_theme["glow"].split("(")[1].split(")")[0].split(",")[:3]
    return f"rgba({r.strip()},{g.strip()},{b.strip()},{alpha})"


def make_starfield(n, width, height, seed):
    rng = random.Random(seed)
    dots = []
    for _ in range(n):
        x, y = rng.randint(0, width), rng.randint(0, height)
        dots.append(f"{x}px {y}px #FFFFFF")
    return ", ".join(dots)


STARS_SMALL = make_starfield(220, 2600, 1800, seed=7)
STARS_MED = make_starfield(70, 2600, 1800, seed=21)
STARS_LARGE = make_starfield(28, 2600, 1800, seed=99)

# ============================================================
# GLOBAL CSS — fonts, starfield, nebula, HUD chrome, telescope
# ============================================================
starfield_css = f"""
.starfield-layer {{ position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden; }}
.starfield-layer span {{ position: absolute; border-radius: 50%; background: #fff; }}
#stars-sm, #stars-sm::after {{
    content: ""; position: absolute; top: 0; left: 0; width: 2px; height: 2px;
    background: transparent; box-shadow: {STARS_SMALL}; animation: twinkleA 6s ease-in-out infinite alternate;
}}
#stars-md, #stars-md::after {{
    content: ""; position: absolute; top: 0; left: 0; width: 3px; height: 3px;
    background: transparent; box-shadow: {STARS_MED}; opacity: 0.85;
    animation: twinkleB 4.5s ease-in-out infinite alternate;
}}
#stars-lg, #stars-lg::after {{
    content: ""; position: absolute; top: 0; left: 0; width: 3px; height: 3px; border-radius: 50%;
    background: transparent; box-shadow: {STARS_LARGE}; opacity: 0.95;
    animation: twinkleC 3.2s ease-in-out infinite alternate;
}}
@keyframes twinkleA {{ from {{ opacity: 0.25; }} to {{ opacity: 0.9; }} }}
@keyframes twinkleB {{ from {{ opacity: 0.4; }} to {{ opacity: 1; }} }}
@keyframes twinkleC {{ from {{ opacity: 0.5; }} to {{ opacity: 1; }} filter: drop-shadow(0 0 3px #fff); }}
""" if show_starfield else ""

_global_css_html = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=JetBrains+Mono:wght@400;600;700&display=swap');

    html, body, [class*="css"] {{ font-family: 'JetBrains Mono', 'Courier New', monospace; }}

    .stApp {{
        background:
            radial-gradient(ellipse 900px 600px at 12% 8%, {active_theme['nebula1']}, transparent 60%),
            radial-gradient(ellipse 800px 700px at 88% 85%, {active_theme['nebula2']}, transparent 60%),
            radial-gradient(ellipse 1200px 900px at 50% 50%, #0a0d16 0%, {active_theme['bg']} 70%);
        background-attachment: fixed;
        color: #C5D1EC;
        animation: nebulaDrift 40s ease-in-out infinite alternate;
    }}
    @keyframes nebulaDrift {{
        0% {{ background-position: 0% 0%, 0% 0%, 0% 0%; }}
        100% {{ background-position: 4% 3%, -3% -4%, 0% 0%; }}
    }}

    {starfield_css}
    section[data-testid="stSidebar"] > div {{ position: relative; z-index: 2; }}
    .main .block-container {{ position: relative; z-index: 2; }}

    h1, h2, h3 {{
        color: {active_theme['accent']} !important;
        font-family: 'Orbitron', 'Courier New', monospace;
        letter-spacing: 1.5px;
        text-shadow: 0 0 12px {active_theme['glow']};
    }}

    /* --- Title with shimmer sweep --- */
    .cah-title {{
        font-family: 'Orbitron', monospace; font-weight: 900; font-size: 2.6rem;
        letter-spacing: 3px; margin-bottom: 0;
        background: linear-gradient(100deg, {active_theme['accent']} 20%, #ffffff 45%, {active_theme['accent2']} 60%, {active_theme['accent']} 80%);
        background-size: 220% auto; -webkit-background-clip: text; background-clip: text;
        color: transparent; animation: shimmer 5s linear infinite;
        text-shadow: 0 0 18px {active_theme['glow']};
    }}
    @keyframes shimmer {{ to {{ background-position: -220% center; }} }}
    .cah-title::after {{ content: "_"; animation: blink 1s steps(1) infinite; color: {active_theme['accent']}; }}
    @keyframes blink {{ 50% {{ opacity: 0; }} }}
    .cah-subtitle {{ letter-spacing: 4px; opacity: 0.75; font-size: 0.85rem; margin-top: -4px; }}

    .status-pill {{
        display: inline-flex; align-items: center; gap: 6px; font-size: 0.75rem;
        border: 1px solid {active_theme['glow']}; border-radius: 999px; padding: 3px 10px;
        background: rgba(255,255,255,0.03); letter-spacing: 1px;
    }}
    .status-dot {{ width: 8px; height: 8px; border-radius: 50%; background: #37ff8b;
        box-shadow: 0 0 8px #37ff8b; animation: pulseDot 1.6s ease-in-out infinite; }}
    @keyframes pulseDot {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.35; }} }}

    /* --- Section titles with corner-bracket HUD style --- */
    .hud-title {{
        display: flex; align-items: center; gap: 10px; margin: 6px 0 2px 0;
        font-family: 'Orbitron', monospace; font-weight: 700; letter-spacing: 2px;
        color: {active_theme['accent']}; text-shadow: 0 0 10px {active_theme['glow']};
        font-size: 1.15rem;
    }}
    .hud-title .bracket {{ color: {active_theme['accent2']}; opacity: 0.7; }}
    .hud-rule {{
        height: 2px; margin: 4px 0 18px 0; border-radius: 2px;
        background: linear-gradient(90deg, {active_theme['accent']}, transparent 75%);
        box-shadow: 0 0 8px {active_theme['glow']};
    }}

    /* --- HUD panel wrapper (adds a faint console frame around a block) --- */
    .hud-panel {{
        position: relative; border: 1px solid {_glow(0.35)}; border-radius: 10px;
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00));
        padding: 18px 18px 8px 18px; margin-bottom: 18px;
    }}
    .hud-panel::before, .hud-panel::after {{
        content: ""; position: absolute; width: 14px; height: 14px; border-color: {active_theme['accent']};
    }}
    .hud-panel::before {{ top: -1px; left: -1px; border-top: 2px solid; border-left: 2px solid; }}
    .hud-panel::after {{ bottom: -1px; right: -1px; border-bottom: 2px solid; border-right: 2px solid; }}

    /* --- Telescope Eyepiece Lens Frame --- */
    .telescope-wrapper {{
        position: relative; width: 320px; height: 320px; margin: 15px auto;
        border-radius: 50%; border: 4px solid {active_theme['accent']};
        box-shadow: 0 0 35px {active_theme['glow']}, inset 0 0 30px rgba(0, 0, 0, 0.95),
                    0 0 0 6px rgba(255,255,255,0.03), 0 0 0 8px {_glow(0.25)};
        overflow: hidden; background-color: #000;
        animation: focusIn 0.9s cubic-bezier(.2,.8,.3,1) both, apertureIris 0.7s ease-out both;
    }}
    @keyframes focusIn {{ from {{ filter: blur(14px); opacity: 0.2; }} to {{ filter: blur(0); opacity: 1; }} }}
    @keyframes apertureIris {{ from {{ clip-path: circle(0% at 50% 50%); }} to {{ clip-path: circle(75% at 50% 50%); }} }}

    /* Corner targeting brackets around every telescope viewport */
    .telescope-outer {{ position: relative; width: 320px; margin: 0 auto 6px auto; }}
    .telescope-outer::before, .telescope-outer::after,
    .telescope-outer .tb-tr, .telescope-outer .tb-bl {{
        content: ""; position: absolute; width: 22px; height: 22px; border-color: {active_theme['accent2']};
        opacity: 0.85; z-index: 5;
    }}
    .telescope-outer::before {{ top: -6px; left: -6px; border-top: 3px solid; border-left: 3px solid; }}
    .telescope-outer::after {{ bottom: -6px; right: -6px; border-bottom: 3px solid; border-right: 3px solid; }}

    .telescope-img {{
        display: block; width: 100% !important; height: 320px !important;
        object-fit: cover !important; border-radius: 50% !important;
        transition: transform 0.5s ease;
    }}
    .telescope-wrapper:hover .telescope-img {{ transform: scale(1.06); }}

    .telescope-reticle {{
        position: absolute; top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none; z-index: 10; border-radius: 50%;
        background:
            radial-gradient(circle, transparent 48%, rgba(0, 0, 0, 0.8) 82%, rgba(0, 0, 0, 0.98) 100%),
            linear-gradient(to right, transparent 49.3%, {_glow(reticle_opacity)} 49.7%, {_glow(reticle_opacity)} 50.3%, transparent 50.7%),
            linear-gradient(to bottom, transparent 49.3%, {_glow(reticle_opacity)} 49.7%, {_glow(reticle_opacity)} 50.3%, transparent 50.7%);
    }}
    .telescope-reticle::after {{
        content: ""; position: absolute; inset: 14%; border-radius: 50%;
        border: 1px dashed {_glow(reticle_opacity * 0.8)};
        animation: slowSpin 18s linear infinite;
    }}
    @keyframes slowSpin {{ to {{ transform: rotate(360deg); }} }}
    .lens-label {{
        text-align: center; font-size: 0.72rem; letter-spacing: 2px; opacity: 0.65;
        margin-top: -4px; text-transform: uppercase;
    }}

    /* --- Radar / scanning sweep shown while inference runs --- */
    .radar-wrap {{ width: 220px; height: 220px; margin: 10px auto; position: relative; }}
    .radar-circle {{
        width: 100%; height: 100%; border-radius: 50%; position: relative;
        border: 2px solid {_glow(0.5)}; overflow: hidden;
        background: repeating-radial-gradient(circle, {_glow(0.08)} 0, transparent 2px, transparent 22px);
    }}
    .radar-sweep {{
        position: absolute; inset: 0; border-radius: 50%;
        background: conic-gradient(from 0deg, {_glow(0.9)}, transparent 35%);
        animation: sweepRotate 1.4s linear infinite;
    }}
    @keyframes sweepRotate {{ to {{ transform: rotate(360deg); }} }}
    .radar-caption {{
        text-align: center; margin-top: 10px; letter-spacing: 2px; font-size: 0.8rem;
        color: {active_theme['accent']};
    }}
    .radar-caption::after {{ content: "..."; animation: ellipsis 1.2s steps(4) infinite; }}
    @keyframes ellipsis {{ 0% {{ content: "."; }} 33% {{ content: ".."; }} 66% {{ content: "..."; }} }}

    /* --- Alert badge --- */
    .alert-badge {{
        display: inline-block; padding: 6px 16px; border-radius: 6px; font-weight: 700;
        letter-spacing: 2px; font-family: 'Orbitron', monospace; font-size: 0.95rem;
        border: 1px solid currentColor;
    }}
    .alert-Low {{ color: #37ff8b; box-shadow: 0 0 10px rgba(55,255,139,0.3); }}
    .alert-Moderate {{ color: #ffcc33; box-shadow: 0 0 10px rgba(255,204,51,0.3); }}
    .alert-High {{ color: #ff3b5c; box-shadow: 0 0 14px rgba(255,59,92,0.55); animation: alertPulse 1.1s ease-in-out infinite; }}
    @keyframes alertPulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.55; }} }}

    /* --- Circular anomaly gauge --- */
    .gauge {{
        width: 150px; height: 150px; border-radius: 50%; margin: 6px auto;
        display: flex; align-items: center; justify-content: center; position: relative;
    }}
    .gauge-inner {{
        width: 116px; height: 116px; border-radius: 50%; background: {active_theme['bg']};
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        border: 1px solid {_glow(0.4)};
    }}
    .gauge-value {{ font-family: 'Orbitron', monospace; font-size: 1.3rem; color: {active_theme['accent']}; }}
    .gauge-label {{ font-size: 0.62rem; letter-spacing: 1.5px; opacity: 0.7; margin-top: 2px; }}

    div[data-testid="stMetric"] {{
        background: rgba(16, 23, 38, 0.55); border: 1px solid {active_theme['glow']};
        border-radius: 8px; padding: 10px 15px; backdrop-filter: blur(2px);
    }}
    div[data-testid="stMetricValue"] {{ color: {active_theme['accent']} !important; font-family: 'Orbitron', monospace; }}

    div[data-testid="stTabs"] button {{ font-family: 'Orbitron', monospace; letter-spacing: 1px; }}
    div[data-testid="stTabs"] button[aria-selected="true"] {{
        color: {active_theme['accent']} !important; border-bottom-color: {active_theme['accent']} !important;
    }}

    .stProgress > div > div > div > div {{ background-color: {active_theme['accent']}; }}

    ::-webkit-scrollbar {{ width: 10px; }}
    ::-webkit-scrollbar-thumb {{ background: {active_theme['glow']}; border-radius: 6px; }}
    </style>
    <div class="starfield-layer"><div id="stars-sm"></div><div id="stars-md"></div><div id="stars-lg"></div></div>
    """
st.html(textwrap.dedent(_global_css_html))


def section_title(icon, text):
    st.html(
        f'<div class="hud-title"><span class="bracket">◤</span>{icon} {text} '
        f'<span class="bracket">◢</span></div><div class="hud-rule"></div>'
    )


def _to_data_uri(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def telescope_view(pil_or_array, caption, use_reticle=True):
    data_uri = _to_data_uri(pil_or_array)
    reticle_div = '<div class="telescope-reticle"></div>' if use_reticle else ""
    block = (
        '<div class="telescope-outer">'
        '<div class="telescope-wrapper">'
        f'<img class="telescope-img" src="{data_uri}" />'
        f"{reticle_div}"
        "</div></div>"
        f'<div class="lens-label">{caption}</div>'
    )
    st.html(block)


def anomaly_gauge(score, level, max_scale=3000):
    pct = max(0.0, min(1.0, score / max_scale))
    color = {"Low": "#37ff8b", "Moderate": "#ffcc33", "High": "#ff3b5c"}[level]
    deg = pct * 360
    gauge_html = f"""<div class="gauge" style="background: conic-gradient({color} {deg}deg, rgba(255,255,255,0.08) {deg}deg);">
<div class="gauge-inner">
<div class="gauge-value">{score:.0f}</div>
<div class="gauge-label">ANOMALY UNITS</div>
</div>
</div>"""
    st.html(gauge_html)


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


def cam_overlay(image, heatmap, alpha):
    original = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255
    colour = np.zeros_like(original)
    colour[..., 0] = heatmap
    colour[..., 1] = 0.20 * (1 - heatmap)
    colour[..., 2] = 1 - heatmap
    blended = (1 - alpha) * original + alpha * colour
    return (255 * np.clip(blended, 0, 1)).astype(np.uint8)


def get_anomaly_level(score):
    if score >= 2500:
        return "High"
    if score >= 1000:
        return "Moderate"
    return "Low"


# ============================================================
# HEADER
# ============================================================
st.html('<div class="cah-title">COSMIC ANOMALY HUNTER</div>')
st.html('<div class="cah-subtitle">OBSERVATORY TARGETING SYSTEM · MORPHOLOGY &amp; ANOMALY ANALYSIS</div>')
st.html(
    '<div style="margin: 10px 0 6px 0;"><span class="status-pill">'
    '<span class="status-dot"></span>ARRAY ONLINE</span>&nbsp; '
    f'<span class="status-pill">DEVICE: {DEVICE.type.upper()}</span>&nbsp; '
    f'<span class="status-pill">THEME: {theme_choice.upper()}</span></div>'
)
st.write("")

st.session_state.setdefault("mission_log", [])

if uploaded_file is None:
    st.info("🛰️ Awaiting image target from sidebar control panel...")
    with st.expander("ℹ️ About Cosmic Anomaly Hunter", expanded=True):
        st.markdown(
            """
**Cosmic Anomaly Hunter** is a deep-space triage console for galaxy imagery. Point it at a
cutout image and it runs two neural networks in tandem:

1. **Morphology Classifier** — a ResNet50 fine-tuned to sort galaxies into
   *Elliptical*, *Spiral*, or *Merger / Other*, with **Grad-CAM** highlighting the
   pixels that drove the decision.
2. **Anomaly Detector** — a convolutional **autoencoder** trained to reconstruct
   ordinary galaxies. Targets it reconstructs poorly (high MSE) are flagged as
   visually unusual — potential mergers, artifacts, or genuinely rare objects
   worth a second look.

The goal is quick, explainable **anomaly triage**: help a human reviewer decide,
in seconds, which images in a large survey deserve closer attention.
            """
        )
    st.stop()

if not Path(classifier_path).is_file() or not Path(autoencoder_path).is_file():
    st.error("⚠️ Model weight files missing. Check sidebar configuration.")
    st.stop()

raw_image = Image.open(uploaded_file).convert("RGB")
image = ImageEnhance.Brightness(raw_image).enhance(exposure_val)
image = ImageEnhance.Contrast(image).enhance(contrast_val)

classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
autoencoder_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# --- Scanning animation while inference runs ---
scan_placeholder = st.empty()
with scan_placeholder.container():
    radar_html = """<div class="radar-wrap">
<div class="radar-circle"><div class="radar-sweep"></div></div>
</div>
<div class="radar-caption">ACQUIRING TELESCOPE ALIGNMENT &amp; RUNNING NEURAL INFERENCE</div>"""
    st.html(radar_html)
    scan_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.006)
        scan_bar.progress(percent_complete + 1)
scan_placeholder.empty()

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
overlay = cam_overlay(image, grad_cam(classifier, classifier_tensor, predicted_index), heatmap_alpha)

rec_img_np = reconstruction[0].cpu().permute(1, 2, 0).numpy()
rec_img_np = (np.clip(rec_img_np, 0, 1) * 255).astype(np.uint8)
rec_img_pil = Image.fromarray(rec_img_np)

# Log this scan to the session's mission log (dedup on rerun via slider tweaks is fine — informative history)
st.session_state["mission_log"].append(
    {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Target": uploaded_file.name,
        "Class": predicted_class,
        "Confidence": f"{confidence:.1f}%",
        "Anomaly": f"{anomaly_score:.1f}",
        "Alert": anomaly_level,
    }
)

# ============================================================
# TABBED CONSOLE
# ============================================================
tab_telemetry, tab_optical, tab_scan, tab_analysis, tab_log, tab_about = st.tabs(
    ["🛰️ Telemetry", "🔭 Optical Feed", "🌀 Anomaly Scan", "📈 Analysis", "🗂️ Mission Log", "ℹ️ About"]
)

with tab_telemetry:
    section_title("📊", "TELEMETRY SUMMARY")
    with st.container(border=True):
        left, right = st.columns([2, 1])
        with left:
            m1, m2 = st.columns(2)
            m1.metric("Target Morphology", predicted_class)
            m2.metric("Signal Confidence", f"{confidence:.1f}%")
            st.html(
                f'<div style="margin-top:10px;">ALERT LEVEL &nbsp; '
                f'<span class="alert-badge alert-{anomaly_level}">{anomaly_level.upper()}</span></div>'
            )
        with right:
            anomaly_gauge(anomaly_score, anomaly_level)

with tab_optical:
    section_title("🔭", "MAIN OPTICAL EYEPIECE VIEWERS")
    img_col, cam_col = st.columns(2, gap="large")
    with img_col:
        telescope_view(image, "DIRECT OPTICAL LENS", use_reticle=show_reticle)
    with cam_col:
        telescope_view(overlay, "THERMAL / GRAD-CAM TARGET LOCK", use_reticle=show_reticle)

with tab_scan:
    section_title("🔄", "AUTOENCODER ANOMALY VERIFICATION")
    orig_col, rec_col = st.columns(2, gap="large")
    with orig_col:
        telescope_view(image.resize((128, 128)), "RAW FEED (128×128)", use_reticle=show_reticle)
    with rec_col:
        telescope_view(rec_img_np, "RECONSTRUCTION LENS", use_reticle=show_reticle)
        st.caption(f"Raw Reconstruction MSE: `{raw_mse:.6f}`")

    st.html("<br>")
    section_title("🎚️", "ANOMALY REVEAL SLIDER")
    st.caption(
        "Drag to blend from the raw feed into the autoencoder's reconstruction — "
        "regions that resist blending cleanly are where the reconstruction disagrees most."
    )
    reveal = st.slider("Reconstruction blend", 0, 100, 50, label_visibility="collapsed")
    base = np.asarray(image.resize((128, 128)), dtype=np.float32)
    recon_up = np.asarray(rec_img_pil.resize((128, 128)), dtype=np.float32)
    t = reveal / 100
    blended = ((1 - t) * base + t * recon_up).clip(0, 255).astype(np.uint8)
    telescope_view(blended, f"BLEND {reveal}% RECONSTRUCTION", use_reticle=show_reticle)

with tab_analysis:
    prob_col, detail_col = st.columns(2, gap="large")
    with prob_col:
        section_title("📈", "MORPHOLOGY PROBABILITIES")
        for class_name, prob in zip(CLASS_NAMES, probabilities):
            pct = float(prob * 100)
            st.write(f"**{class_name}** — `{pct:.1f}%`")
            st.progress(int(round(pct)))

    with detail_col:
        section_title("🧠", "DEEP INTELLIGENCE ANALYSIS")
        st.write(
            f"The primary target structure is classified as **{predicted_class}** morphology. "
            f"{CLASS_DESCRIPTIONS[predicted_class]} System confidence is evaluated at **{confidence:.1f}%**."
        )
        st.write(
            f"Reconstruction error registers **{anomaly_score:.2f}** anomaly units, "
            f"placing this target in the **{anomaly_level}** alert band."
        )

    st.html("<br>")
    section_title("⚙️", "SENSOR METADATA")
    st.json({
        "Classifier Engine": "ResNet50",
        "Anomaly Detector": "Convolutional Autoencoder",
        "Explainability": "Grad-CAM Activation Visualizer",
        "Spectrum Palette": theme_choice,
        "Hardware Unit": DEVICE.type.upper(),
        "Target File": html.escape(uploaded_file.name),
    })

with tab_log:
    section_title("🗂️", "MISSION LOG")
    st.caption("Every target acquired this session, most recent first.")
    log_rows = list(reversed(st.session_state["mission_log"]))
    st.dataframe(log_rows, use_container_width=True, hide_index=True)
    if st.button("🗑️ Clear Mission Log"):
        st.session_state["mission_log"] = []
        st.rerun()

with tab_about:
    section_title("ℹ️", "ABOUT THIS PROJECT")
    st.markdown(
        """
**Cosmic Anomaly Hunter** is a deep-space triage console built around two neural
networks working side by side on the same galaxy image:

- 🧭 **Morphology Classifier (ResNet50):** sorts each target into *Elliptical*,
  *Spiral*, or *Merger / Other*, with a **Grad-CAM** heatmap showing exactly which
  pixels the network weighed most heavily.
- 🌀 **Anomaly Detector (Convolutional Autoencoder):** learns to compress and
  reconstruct *typical* galaxy images. Targets that come back blurry or wrong —
  high reconstruction error — are flagged as visually unusual and surfaced for
  human review via the alert badge and anomaly gauge.

**Why it matters:** modern sky surveys produce galaxy cutouts far faster than any
team can eyeball them by hand. Cosmic Anomaly Hunter is meant to sit in front of
that firehose — pairing a fast, explainable classification with an anomaly score,
so a reviewer can jump straight to the handful of images that look genuinely odd:
mergers, imaging artifacts, or objects that don't fit the mould at all.

**Pipeline at a glance:** upload → optics adjustment (exposure / contrast) →
ResNet50 classification + Grad-CAM → autoencoder reconstruction → anomaly scoring
→ HUD telemetry and mission log.
        """
    )