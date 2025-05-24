import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- Cài đặt thiết bị ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Load model pretrained ---
@st.cache_resource
def load_model():
    model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    return model

model = load_model()

# --- Giá trị mean và std để normalize theo ImageNet ---
NORM_MEAN = np.array([0.485, 0.456, 0.406])
NORM_STD = np.array([0.229, 0.224, 0.225])

plain_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# --- Hàm hiển thị ảnh ---
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

# --- Hàm FGSM ---
def fast_gradient_sign_method(model, imgs, labels, epsilon=0.02):
    imgs = imgs.clone().detach().to(device).requires_grad_(True)
    output = model(imgs)
    loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(output, dim=1), labels.to(device))
    model.zero_grad()
    loss.backward()
    grad_sign = imgs.grad.data.sign()
    adv_imgs = imgs + epsilon * grad_sign
    # Clamp theo normalize
    for c in range(3):
        adv_imgs[:, c, :, :] = torch.clamp(adv_imgs[:, c, :, :], (0 - NORM_MEAN[c]) / NORM_STD[c], (1 - NORM_MEAN[c]) / NORM_STD[c])
    return adv_imgs.detach(), grad_sign

# --- App ---
st.set_page_config(page_title="FGSM Attack Demo", layout="centered")
st.title("⚔️ Demo Tấn công Adversarial (FGSM) trên ResNet34")

uploaded_file = st.file_uploader("📤 Upload một ảnh JPG/PNG", type=["jpg", "jpeg", "png"])
epsilon = st.slider("⚙️ Chọn epsilon (mức độ nhiễu)", 0.0, 0.2, 0.02, 0.01)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Ảnh gốc", use_container_width=True)

    input_tensor = plain_transforms(img).unsqueeze(0).to(device)

    # Dự đoán ảnh gốc
    with torch.no_grad():
        output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    st.write(f"📌 Dự đoán ảnh gốc: **Class ID = {pred_class}**")

    # Gán nhãn giả để tấn công (dùng chính nhãn mô hình dự đoán)
    label = torch.tensor([pred_class]).to(device)

    # Tạo ảnh nhiễu
    adv_img, noise = fast_gradient_sign_method(model, input_tensor, label, epsilon)

    st.write("🖼️ Ảnh sau khi thêm nhiễu:")
    imshow(adv_img[0])

    # Dự đoán lại trên ảnh nhiễu
    with torch.no_grad():
        adv_output = model(adv_img)
    adv_class = adv_output.argmax(dim=1).item()

    st.write(f"📌 Dự đoán ảnh đã nhiễu: **Class ID = {adv_class}**")

    # Đưa ra nhận xét
    if pred_class != adv_class:
        st.error("💥 Mô hình đã bị đánh lừa! Dự đoán đã thay đổi.")
    else:
        st.success("✅ Mô hình vẫn nhận diện đúng dù đã bị tấn công.")
