import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import time
import math
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps

st.set_page_config(page_title="Mehek's AI Projects", layout="wide")

# --- Personal Header ---
st.title("👋 Welcome to Mehek's Personal Site")
st.write("""
Hi! I'm Mehek. I work on machine learning and explainability techniques.  
This demo showcases a CNN trained on MNIST and uses **PEEK** to explain its predictions.
""")


# --- CNN Model ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        self.activations = []
        x = self.conv1(x)
        self.activations.append(x.clone().detach())
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        self.activations.append(x.clone().detach())
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        self.activations.append(x.clone().detach())
        x = self.relu3(x)

        x = self.fc2(x)
        return x


# Load model
model = SimpleCNN()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# --- Choose input type ---
input_mode = st.radio(
    "Choose input type:", ["Draw your own digit ✍️", "Use MNIST test sample"]
)


# --- Image preparation ---
def process_image(pil_img):
    pil_img = ImageOps.invert(pil_img.convert("L")).resize((28, 28))
    img_arr = np.array(pil_img) / 255.0
    img_tensor = torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    img_tensor = (img_tensor - 0.5) / 0.5  # normalize like training
    return img_tensor, pil_img


if input_mode == "Draw your own digit ✍️":
    st.subheader("🖌️ Draw below")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        drawn_img = Image.fromarray(
            (canvas_result.image_data[:, :, 0]).astype(np.uint8)
        )
        img_tensor, display_img = process_image(drawn_img)

else:
    # Use sample from MNIST test set
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    mnist_test = MNIST(root="./data", train=False, download=True, transform=transform)
    example_img_tensor, true_label = mnist_test[0]
    display_img = (example_img_tensor * 0.5 + 0.5).squeeze().numpy()
    img_tensor = example_img_tensor.unsqueeze(0)

# --- Predict ---
with torch.no_grad():
    output = model(img_tensor)
    prediction = output.argmax(dim=1).item()

st.subheader("🧠 Prediction Result")
if input_mode == "Draw your own digit ✍️":
    st.image(display_img, caption=f"Predicted: {prediction}", width=120)
else:
    st.image(
        display_img,
        caption=f"Predicted: {prediction} | True Label: {true_label}",
        width=120,
    )


# --- PEEK: Generate Heatmaps ---
def compute_peek_heatmaps(activations):
    heatmaps = []
    for act in activations:
        if act.dim() == 4:
            avg = act.squeeze(0).mean(0)
        elif act.dim() == 2:
            vec = act.squeeze(0)
            length = vec.numel()
            side = math.isqrt(length)
            if side * side == length:
                avg = vec.reshape(side, side)
            else:
                new_size = (side + 1) * (side + 1)
                pad = torch.nn.functional.pad(vec, (0, new_size - length))
                avg = pad.reshape(side + 1, side + 1)
        else:
            continue
        norm = (avg - avg.min()) / (avg.max() - avg.min() + 1e-5)
        heatmaps.append(norm.numpy())
    return heatmaps


heatmaps = compute_peek_heatmaps(model.activations)

# --- PEEK Viewer ---
st.subheader("🔍 PEEK Explainability")
st.write("Layer-by-layer heatmaps showing what the model focuses on internally.")

col1, col2 = st.columns([1, 3])
with col1:
    animate = st.button("▶️ Play Animation")
    delay = st.slider("Speed (seconds)", 0.1, 1.0, 0.5, step=0.1)

img_container = col2.empty()

if animate:
    for i in range(len(heatmaps)):
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.imshow(heatmaps[i], cmap="viridis")
        ax.set_title(f"Layer {i}")
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        buf.seek(0)
        img_container.image(buf, use_container_width=False)
        plt.close(fig)
        time.sleep(delay)
else:
    selected = st.slider("Layer", 0, len(heatmaps) - 1, 0)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.imshow(heatmaps[selected], cmap="hot")
    ax.set_title(f"Layer {selected}")
    ax.axis("off")
    st.pyplot(fig)
