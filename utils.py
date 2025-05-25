import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Normalization constants (from ImageNet)
NORM_MEAN = np.array([0.485, 0.456, 0.406])
NORM_STD = np.array([0.229, 0.224, 0.225])

# Load label list
try:
    import json
    with open("label_list.json", "r") as f:
        label_names = json.load(f)
except FileNotFoundError:
    label_names = [f"Class {i}" for i in range(1000)]  # fallback

def load_model():
    model = torchvision.models.resnet34(weights="IMAGENET1K_V1")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def fast_gradient_sign_method(model, imgs, labels, epsilon=0.02):
    inp_imgs = imgs.clone().detach().requires_grad_(True)
    preds = model(inp_imgs)
    preds_log = F.log_softmax(preds, dim=-1)
    loss = -torch.gather(preds_log, 1, labels.unsqueeze(-1)).sum()
    loss.backward()

    grad_sign = torch.sign(inp_imgs.grad)
    adv_imgs = imgs + epsilon * grad_sign
    adv_imgs = torch.clamp(adv_imgs, 0, 1)
    return adv_imgs.detach(), grad_sign.detach()

def show_prediction_plot(img, pred, adv_img=None, adv_pred=None, noise=None, K=5):
    def tensor_to_img(tensor):
        img = tensor.detach().cpu().permute(1, 2, 0).numpy()
        img = (img * NORM_STD[None, None]) + NORM_MEAN[None, None]
        img = np.clip(img, 0, 1)
        return img

    fig_cols = 5 if (adv_img is not None and noise is not None) else 2
    fig, ax = plt.subplots(1, fig_cols, figsize=(12, 2))

    # Lấy label top-1
    orig_label = label_names[pred.argmax().item()] if pred is not None else "Unknown"
    adv_label = label_names[adv_pred.argmax().item()] if adv_pred is not None else "Unknown"

    # Ảnh gốc
    ax[0].imshow(tensor_to_img(img))
    ax[0].set_title(f"Original\n[{orig_label}]")
    ax[0].axis("off")

    if adv_img is not None and noise is not None:
        # Ảnh đối kháng
        ax[1].imshow(tensor_to_img(adv_img))
        ax[1].set_title(f"Adversarial\n[{adv_label}]")
        ax[1].axis("off")

        # Noise
        noise_vis = noise.permute(1, 2, 0).detach().cpu().numpy()
        noise_vis = noise_vis * 0.5 + 0.5  # scale to [0,1]
        noise_vis = np.clip(noise_vis, 0, 1)
        ax[2].imshow(noise_vis)
        ax[2].set_title("Noise")
        ax[2].axis("off")

        ax[3].axis("off")  # spacing

    # Top-K dự đoán (gốc hoặc đối kháng)
    output = adv_pred if adv_pred is not None else pred
    topk_vals, topk_idx = torch.topk(torch.softmax(output, dim=-1), K)
    topk_vals = topk_vals.detach().cpu().numpy()
    topk_idx = topk_idx.detach().cpu().numpy()
    labels = [label_names[i] if i < len(label_names) else str(i) for i in topk_idx]

    ax[-1].barh(np.arange(K), topk_vals * 100, color="skyblue")
    ax[-1].set_yticks(np.arange(K))
    ax[-1].set_yticklabels(labels)
    ax[-1].invert_yaxis()
    ax[-1].set_xlabel("Confidence (%)")
    ax[-1].set_title("Top-5 Predictions")

    st.pyplot(fig)
