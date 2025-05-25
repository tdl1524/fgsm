import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

from utils import (
    load_model,
    fast_gradient_sign_method,
    show_prediction_plot,
    NORM_MEAN,
    NORM_STD
)

st.set_page_config(page_title="FGSM Adversarial Demo", layout="wide")
st.title("🔒 FGSM Adversarial Attack on ResNet34")
st.markdown("""
Upload an image to see how a small perturbation (using **Fast Gradient Sign Method**) can mislead a deep learning model.
""")

# Upload image
uploaded_img = st.file_uploader("Upload an image (JPG/PNG):", type=["jpg", "jpeg", "png"])

# Epsilon slider
epsilon = st.sidebar.slider("Attack Strength (ε)", 0.0, 0.2, 0.02, step=0.005)
st.sidebar.markdown("Higher ε → more visible noise, stronger attack.")

# Load model once
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# Prepare transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

if uploaded_img is not None:
    image = Image.open(uploaded_img).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)
    label = model(img_tensor).argmax(dim=1)

    # Predict original
    with torch.no_grad():
        pred = model(img_tensor)

    # Run FGSM attack
    adv_img_tensor, noise = fast_gradient_sign_method(model, img_tensor, label, epsilon=epsilon)

    # Predict adversarial
    with torch.no_grad():
        adv_pred = model(adv_img_tensor)

    # Display comparison
    show_prediction_plot(
        img_tensor[0],
        pred[0],
        adv_img=adv_img_tensor[0],
        adv_pred=adv_pred[0],
        noise=noise[0]
    )
