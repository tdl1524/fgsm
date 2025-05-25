import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from utils import (
    load_model, show_prediction_plot, fast_gradient_sign_method, NORM_MEAN, NORM_STD
)

st.title("FGSM Adversarial Attack on ResNet34")

uploaded_img = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    image = Image.open(uploaded_img).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    # Load model and transform
    model = load_model()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    img_tensor = transform(image).unsqueeze(0)

    # Predict original
    with torch.no_grad():
        pred_orig = model(img_tensor)

    # Adversarial attack
    adv_img_tensor, noise = fast_gradient_sign_method(model, img_tensor, pred_orig.argmax(dim=1), epsilon=0.02)
    with torch.no_grad():
        pred_adv = model(adv_img_tensor)

    # Visualization
    show_prediction_plot(img_tensor[0], pred_orig[0], adv_img_tensor[0], pred_adv[0], noise[0])
