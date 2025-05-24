import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import ImageNet
from torchvision.models import resnet34
from torchvision.models._api import Weights

# 🛠 Cấu hình page PHẢI đặt trước mọi thứ khác
st.set_page_config(page_title="FGSM Attack Demo", layout="centered")

# --- Thiết bị ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Tải model pretrained và labels ---
@st.cache_resource
def load_model():
    weights = Weights.IMAGENET1K_V1
    model = resnet34(weights=weights)
    model.eval()
    model.to(device)
    return model, weights.meta["categories"]

model, imagenet_classes = load_model()

# --- Chuẩn hóa ảnh ---
NORM_MEAN = np.array([0.485, 0.456, 0.406])
NORM_STD = np.array([0.229, 0.224, 0.225])

plain_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# --- Hiển thị ảnh ---
def imshow(img_tensor, title=""):
    img = img_tensor.cpu().permute(1, 2, 0).numpy()
    img = img * NORM_STD + NORM_MEAN
    img = np.clip(img, 0, 1)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')
    if title:
        ax.set_title(title)
    st.pyplot(fig)

# --- FGSM ---
def fast_gradient_sign_method(model, imgs, labels, epsilon=0.02):
    imgs = imgs.clone().detach().to(device).requires_grad_(True)
    output = model(imgs)
    loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(output, dim=1), labels)
    model.zero_grad()
    loss.backward()
    grad_sign = imgs.grad.data.sign()
    adv_imgs = imgs + epsilon * grad_sign
    for c in range(3):
        adv_imgs[:, c, :, :] = torch.clamp(
            adv_imgs[:, c, :, :],
            (0 - NORM_MEAN[c]) / NORM_STD[c],
            (1 - NORM_MEAN[c]) / NORM_STD[c],
        )
    return adv_imgs.detach(), grad_sign

# --- Giao diện Streamlit ---
st.title("⚔️ FGSM Attack Demo on ResNet34")

uploaded_file = st.file_uploader("📤 Upload ảnh JPG/PNG", type=["jpg", "jpeg", "png"])
epsilon = st.slider("⚙️ Chọn epsilon (mức độ nhiễu)", 0.0, 2.0, 0.02, 0.01)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Ảnh gốc", use_container_width=True)

    input_tensor = plain_transforms(img).unsqueeze(0).to(device)

    # --- Dự đoán ảnh gốc ---
    with torch.no_grad():
        output = model(input_tensor)
    pred_class_id = output.argmax(dim=1).item()
    pred_class_name = imagenet_classes[pred_class_id]
    st.markdown(f"🧠 **Dự đoán ảnh gốc:** `{pred_class_name}` (ID: {pred_class_id})")

    label = torch.tensor([pred_class_id]).to(device)

    # --- Tạo ảnh nhiễu ---
    adv_img, noise = fast_gradient_sign_method(model, input_tensor, label, epsilon)

    st.markdown("🖼️ **Ảnh đã thêm nhiễu:**")
    imshow(adv_img[0])

    # --- Dự đoán ảnh đã nhiễu ---
    with torch.no_grad():
        adv_output = model(adv_img)
    adv_class_id = adv_output.argmax(dim=1).item()
    adv_class_name = imagenet_classes[adv_class_id]
    st.markdown(f"🧠 **Dự đoán ảnh đã nhiễu:** `{adv_class_name}` (ID: {adv_class_id})")

    # --- Nhận xét ---
    if pred_class_id != adv_class_id:
        st.error("💥 Mô hình đã bị đánh lừa! Kết quả đã thay đổi.")
    else:
        st.success("✅ Mô hình vẫn nhận diện đúng.")
