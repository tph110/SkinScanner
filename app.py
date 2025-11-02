import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms

# -------------------------------------------------
# 1. Page config
# -------------------------------------------------
st.set_page_config(
    page_title="DermAI – Skin Lesion Classifier",
    page_icon="skin",
    layout="centered",
)

# -------------------------------------------------
# 2. Load model (cached, CPU)
# -------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    # Change the filename if you use .pt instead of .pth
    path = "model.pth"
    model = torch.load(path, map_location=torch.device("cpu"))
    model.eval()
    return model

model = load_model()

# -------------------------------------------------
# 3. Pre-processing (adjust to your training script)
# -------------------------------------------------
# Most skin-lesion models are trained on 224×224 or 256×256.
# Change INPUT_SIZE if your model expects something else.
INPUT_SIZE = 224

preprocess = transforms.Compose([
    transforms.Resize(int(INPUT_SIZE * 1.14)),   # 256 for 224, 292 for 256, etc.
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet stats (common)
        std =[0.229, 0.224, 0.225],
    ),
])

# -------------------------------------------------
# 4. Class names (replace with your own!)
# -------------------------------------------------
# Example for ISIC-2019 (7 classes). Edit to match your training labels.
CLASS_NAMES = [
    "Melanocytic nevus",
    "Melanoma",
    "Benign keratosis",
    "Basal cell carcinoma",
    "Actinic keratosis",
    "Vascular lesion",
    "Dermatofibroma",
]

# -------------------------------------------------
# 5. UI
# -------------------------------------------------
st.title("DermAI – Skin Lesion Classifier")
st.caption("Upload a clear, well-lit skin image (JPG/PNG).")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Max 10 MB",
)

if uploaded_file is not None:
    # ---- Show uploaded image ----
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="Uploaded image", use_column_width=True)

    # ---- Preprocess ----
    input_tensor = preprocess(pil_img).unsqueeze(0)   # (1, C, H, W)

    # ---- Inference (no grad) ----
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze(0)

    # ---- Top-5 (or all if <5 classes) ----
    k = min(5, len(CLASS_NAMES))
    top_vals, top_idx = torch.topk(probs, k=k)
    top_vals = top_vals.tolist()
    top_idx = top_idx.tolist()

    # ---- Results ----
    st.subheader("Prediction")
    for i, (val, idx) in enumerate(zip(top_vals, top_idx), 1):
        label = CLASS_NAMES[idx]
        st.markdown(f"**{i}.** {label} – **{val:.2%}**")

    # ---- Optional bar chart ----
    if st.checkbox("Show probability chart"):
        import pandas as pd
        df = pd.DataFrame({
            "Class": [CLASS_NAMES[i] for i in top_idx],
            "Probability": top_vals,
        })
        st.bar_chart(df.set_index("Class"))

    # ---- Debug info (remove in prod) ----
    if st.checkbox("Show raw probabilities (debug)"):
        st.json({CLASS_NAMES[i]: f"{p:.4%}" for i, p in enumerate(probs.tolist())})

# -------------------------------------------------
# 6. Footer
# -------------------------------------------------
st.caption(
    "Model: `model.pth` | "
    "Python 3.12.2 | "
    "torch 2.9.0 | streamlit 1.39.0"
)
