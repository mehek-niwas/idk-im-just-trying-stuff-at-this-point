import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np

# --- Simple CNN for MNIST ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# Load the model
model = SimpleCNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# UI
st.title("🧠 MNIST Digit Classifier")

uploaded_file = st.file_uploader("Upload a 28x28 grayscale image of a digit", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('L')  # convert to grayscale
    st.image(img, caption="Uploaded Image", width=150)

    # Transform
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    img_tensor = transform(img).unsqueeze(0)  # (1, 1, 28, 28)

    with torch.no_grad():
        output = model(img_tensor)
        prediction = output.argmax(dim=1).item()

    st.success(f"✅ Predicted Digit: **{prediction}**")
