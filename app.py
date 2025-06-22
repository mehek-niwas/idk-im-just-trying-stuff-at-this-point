import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

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

# Load trained model
model = SimpleCNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# --- Get Example from MNIST Test Set ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_test = MNIST(root="./data", train=False, download=True, transform=transform)
example_img_tensor, true_label = mnist_test[0]
example_img = (example_img_tensor * 0.5 + 0.5).squeeze().numpy()  # unnormalize for display
img_tensor = example_img_tensor.unsqueeze(0)  # shape: (1, 1, 28, 28)

# --- Predict ---
with torch.no_grad():
    output = model(img_tensor)
    prediction = output.argmax(dim=1).item()

st.subheader("🖼️ Example Digit and Prediction")
st.image(example_img, caption=f"Predicted: {prediction} | True Label: {true_label}", width=150)

# --- PEEK: Generate Heatmaps ---
def compute_peek_heatmaps(activations):
    heatmaps = []
    for act in activations:
        if act.dim() == 4:  # Conv layers: (B, C, H, W)
            avg = act.squeeze(0).mean(0)  # (H, W)
        elif act.dim() == 2:  # FC layers: (B, N)
            avg = act.squeeze(0).reshape(8, -1)  # visualize as a square-ish map
        else:
            continue
        norm = (avg - avg.min()) / (avg.max() - avg.min() + 1e-5)
        heatmaps.append(norm.numpy())
    return heatmaps

peek_dir = "peek_outputs"
os.makedirs(peek_dir, exist_ok=True)

heatmaps = compute_peek_heatmaps(model.activations)

# --- Heatmap Viewer ---
st.subheader("🔍 PEEK Explainability")
st.write("Layer-by-layer heatmaps showing what the model focuses on internally.")

frame = st.slider("Layer", 0, len(heatmaps) - 1, 0)
fig, ax = plt.subplots()
ax.imshow(heatmaps[frame], cmap="hot")
ax.axis("off")
st.pyplot(fig)
