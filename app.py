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
    img = img.permute(1,2,0).numpy()
    img = (img * NORM_STD) + NORM_MEAN
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    st.pyplot()

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

uploaded_file = st.file_uploader("📷 Tải lên một ảnh JPG/PNG", type=["jpg", "jpeg", "png"])
epsilon = st.slider("⚠️ Chọn cường độ tấn công (epsilon)", 0.0, 0.1, 0.02, 0.005)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh gốc", use_column_width=True)

    # Chuẩn bị ảnh đầu vào
    input_tensor = plain_transforms(image).unsqueeze(0).to(device)

    # Dự đoán nhãn ảnh gốc
    with torch.no_grad():
        output = model(input_tensor)
    pred_label_idx = output.argmax(dim=1).item()
    pred_label_name = imagenet_labels[pred_label_idx]

    st.markdown(f"### ✅ Nhãn dự đoán của ảnh gốc: **{pred_label_name}** (ID: {pred_label_idx})")

    # Gán nhãn để thực hiện FGSM
    true_label = torch.tensor([pred_label_idx]).to(device)

    # Tạo ảnh adversarial
    adv_tensor, noise_tensor = fast_gradient_sign_method(model, input_tensor, true_label, epsilon=epsilon)

    # Dự đoán ảnh adversarial
    with torch.no_grad():
        adv_output = model(adv_tensor)
    adv_label_idx = adv_output.argmax(dim=1).item()
    adv_label_name = imagenet_labels[adv_label_idx]

    # Hiển thị ảnh adversarial
    st.markdown("### 🛠️ Ảnh sau khi thêm nhiễu (adversarial):")
    imshow(adv_tensor[0], title="Ảnh Adversarial")

    # Hiển thị nhiễu
    st.markdown("### 🔍 Nhiễu được thêm vào:")
    imshow(noise_tensor[0] * 0.5 + 0.5, title="Nhiễu (Noise)")

    # Dự đoán trên ảnh bị tấn công
    st.markdown(f"### 🧠 Nhãn dự đoán của ảnh adversarial: **{adv_label_name}** (ID: {adv_label_idx})")

    # So sánh và nhận xét
    if pred_label_idx == adv_label_idx:
        st.warning("⚠️ Mô hình **vẫn nhận diện đúng** sau khi bị tấn công adversarial.")
    else:
        st.success("✅ Mô hình **đã bị đánh lừa** sau khi ảnh bị thêm nhiễu adversarial!")
