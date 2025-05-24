import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from urllib.request import urlopen

# --- Device ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Load model ---
@st.cache_resource
def load_model():
    model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    return model

model = load_model()

# --- Load labels ---
@st.cache_data
def load_imagenet_labels():
    with urlopen("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

imagenet_labels = load_imagenet_labels()

# --- Normalize constants ---
NORM_MEAN = np.array([0.485, 0.456, 0.406])
NORM_STD = np.array([0.229, 0.224, 0.225])

# --- Transform ---
plain_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# --- Hiển thị ảnh ---
def imshow(img_tensor, title=""):
    img = img_tensor.cpu().detach().clone()
    img = img.permute(1, 2, 0).numpy()
    img = (img * NORM_STD) + NORM_MEAN
    img = np.clip(img, 0, 1)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')
    st.pyplot(fig)

# --- FGSM attack ---
def fast_gradient_sign_method(model, imgs, labels, epsilon=0.02):
    imgs = imgs.clone().detach().to(device).requires_grad_(True)
    preds = model(imgs)
    loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(preds, dim=1), labels)
    model.zero_grad()
    loss.backward()
    noise = torch.sign(imgs.grad)
    adv_imgs = imgs + epsilon * noise
    adv_imgs = torch.clamp(adv_imgs, 0, 1).detach()
    return adv_imgs, noise

# --- App ---
st.title("🔍 Phân tích tấn công Adversarial (FGSM) trên ResNet34")

uploaded_file = st.file_uploader("📷
