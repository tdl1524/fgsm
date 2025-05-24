import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# --- Cài đặt device ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Load model pretrained ---
@st.cache_resource
def load_model():
    model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    return model

model = load_model()

# --- Chuẩn bị transform ---
NORM_MEAN = np.array([0.485, 0.456, 0.406])
NORM_STD = np.array([0.229, 0.224, 0.225])
plain_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# --- Hàm hiển thị ảnh ---
def imshow(img_tensor):
    img = img_tensor.cpu().permute(1,2,0).numpy()
    img = (img * NORM_STD) + NORM_MEAN
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis('off')
    st.pyplot()

# --- Hàm FGSM ---
def fast_gradient_sign_method(model, imgs, labels, epsilon=0.02):
    imgs = imgs.clone().detach().to(device).requires_grad_(True)
    preds = model(imgs)
    loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(preds, dim=1), labels.to(device))
    model.zero_grad()
    loss.backward()
    noise = torch.sign(imgs.grad)
    adv_imgs = imgs + epsilon * noise
    adv_imgs = torch.clamp(adv_imgs, 0, 1).detach()
    return adv_imgs, noise

# --- App ---
st.title("Demo Adversarial Attack FGSM trên ResNet34")

uploaded_file = st.file_uploader("Upload một ảnh JPG/PNG", type=["jpg","jpeg","png"])
epsilon = st.slider("Chọn epsilon (cường độ tấn công)", 0.0, 0.1, 0.02, 0.005)

if uploaded_file is not None:
    from PIL import Image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Ảnh gốc", use_column_width=True)

    # Tiền xử lý
    input_tensor = plain_transforms(img).unsqueeze(0).to(device)
    
    # Dự đoán ban đầu
    with torch.no_grad():
        pred = model(input_tensor)
    pred_idx = pred.argmax(dim=1).item()
    st.write(f"Dự đoán của model: Class ID = {pred_idx}")
    
    # Tạo nhãn giả (dùng nhãn dự đoán để tấn công)
    label = torch.tensor([pred_idx]).to(device)
    
    # Tạo ảnh adversarial
    adv_img, noise = fast_gradient_sign_method(model, input_tensor, label, epsilon=epsilon)
    
    st.write("Ảnh adversarial:")
    imshow(adv_img[0])
    
    # Dự đoán adversarial
    with torch.no_grad():
        adv_pred = model(adv_img)
    adv_pred_idx = adv_pred.argmax(dim=1).item()
    st.write(f"Dự đoán trên ảnh adversarial: Class ID = {adv_pred_idx}")
