import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import entr
from scipy.ndimage import zoom
import io
import base64
from PIL import Image, ImageDraw
import time
import os
from collections import defaultdict
from streamlit_drawable_canvas import st_canvas

# Import model architectures
from train_multiple_models import SimpleCNN, DeepCNN, VGGNet, ResNet18, MODEL_CONFIGS

# Set page config
st.set_page_config(
    page_title="CNN PEEK Map Visualizer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .peek-container {
        border: 2px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "feature_maps" not in st.session_state:
    st.session_state.feature_maps = {}

if "current_model" not in st.session_state:
    st.session_state.current_model = None

if "conv_layers" not in st.session_state:
    st.session_state.conv_layers = []

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# PEEK computation functions
def compute_PEEK(feature_maps, h, w):
    positivized_maps = feature_maps + np.abs(np.min(feature_maps))
    entropy_map = -np.sum(entr(positivized_maps), axis=-1)
    zoom_factors = (h / entropy_map.shape[0], w / entropy_map.shape[1])
    peek_map = zoom(entropy_map, zoom_factors, order=1)
    return peek_map


def hook_fn(m, i, o):
    if not st.session_state.current_model.training:
        st.session_state.feature_maps[m] = o


def get_conv_layers(model):
    """Extract convolutional layers from model"""
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(module)
    return conv_layers


def load_model(model_name):
    """Load a trained model"""
    model_path = f"models/{model_name.lower()}_mnist.pth"

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.info("Please run train_multiple_models.py first to train the models.")
        return None

    # Get model class
    model_class = MODEL_CONFIGS[model_name]["model_class"]
    model = model_class()

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Register hooks
    conv_layers = get_conv_layers(model)
    for layer in conv_layers:
        layer.register_forward_hook(hook_fn)

    st.session_state.current_model = model
    st.session_state.conv_layers = conv_layers

    return model, checkpoint


def generate_peek_maps(image_tensor):
    """Generate PEEK maps for an image"""
    if st.session_state.current_model is None:
        return None, None

    # Clear previous feature maps
    st.session_state.feature_maps.clear()

    # Forward pass
    with torch.no_grad():
        _ = st.session_state.current_model(image_tensor)

    # Get original image
    image = image_tensor.squeeze().cpu().numpy()
    h, w = image.shape

    peek_maps = {}

    for layer in st.session_state.conv_layers:
        feature_maps_data = st.session_state.feature_maps.get(layer, None)
        if feature_maps_data is not None:
            feature_maps_data = feature_maps_data[0]  # Access first element
            feature_maps_data = np.moveaxis(feature_maps_data.cpu().numpy(), 0, -1)
            peek_map = compute_PEEK(feature_maps_data, h, w)
            peek_maps[layer] = peek_map

    return peek_maps, image


def create_drawing_canvas():
    """Create a drawing canvas for user input"""
    canvas_result = st_canvas(
        stroke_width=20,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        # Convert to PIL Image
        image = Image.fromarray(canvas_result.image_data)

        # Convert to grayscale and resize to 28x28
        image = image.convert("L")
        image = image.resize((28, 28))

        # Convert to tensor
        image_array = np.array(image)
        image_tensor = torch.FloatTensor(image_array).unsqueeze(0).unsqueeze(0)

        # Normalize
        image_tensor = (image_tensor - 0.5) / 0.5

        return image_tensor, image_array

    return None, None


def plot_peek_maps(image, peek_maps, title="PEEK Maps"):
    """Create a plot with original image and PEEK maps"""
    if not peek_maps:
        return None

    num_layers = len(peek_maps)
    fig, axes = plt.subplots(1, num_layers + 1, figsize=(4 * (num_layers + 1), 4))

    # Original image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # PEEK maps for each layer
    for i, (layer, peek_map) in enumerate(peek_maps.items()):
        axes[i + 1].imshow(image, cmap="gray")
        axes[i + 1].imshow(peek_map, alpha=0.7, cmap="jet")
        axes[i + 1].set_title(f"PEEK - {layer.__class__.__name__}")
        axes[i + 1].axis("off")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    return fig


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    buf.close()
    return img_str


def main():
    # Header
    st.markdown(
        '<h1 class="main-header">🔍 CNN PEEK Map Visualizer</h1>',
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.title("Model Selection")

    # Model selection
    available_models = []
    for model_name in MODEL_CONFIGS.keys():
        model_path = f"models/{model_name.lower()}_mnist.pth"
        if os.path.exists(model_path):
            available_models.append(model_name)

    if not available_models:
        st.sidebar.error("No trained models found!")
        st.sidebar.info(
            "Please run train_multiple_models.py first to train the models."
        )
        return

    selected_model = st.sidebar.selectbox(
        "Choose Model Architecture:", available_models, index=0
    )

    # Load model
    if st.sidebar.button("Load Model") or st.session_state.current_model is None:
        with st.spinner("Loading model..."):
            model, checkpoint = load_model(selected_model)
            if model is not None:
                st.sidebar.success(f"✅ {selected_model} loaded successfully!")

                # Display model info
                st.sidebar.markdown("### Model Information")
                st.sidebar.markdown(f"**Accuracy:** {checkpoint['accuracy']:.2f}%")
                st.sidebar.markdown(
                    f"**Training Time:** {checkpoint['training_time']:.2f}s"
                )
                st.sidebar.markdown(f"**Epochs:** {checkpoint['epochs']}")
                st.sidebar.markdown(f"**Learning Rate:** {checkpoint['lr']}")
                st.sidebar.markdown(
                    f"**Convolutional Layers:** {len(st.session_state.conv_layers)}"
                )

    # Main content
    if st.session_state.current_model is not None:
        # Tabs
        tab1, tab2, tab3 = st.tabs(
            ["🎨 Draw & Predict", "📊 MNIST Test Images", "📈 Model Analysis"]
        )

        with tab1:
            st.header("🎨 Draw Your Own Digit")
            st.markdown("Draw a digit (0-9) in the canvas below and see the PEEK maps!")

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### Drawing Canvas")
                image_tensor, image_array = create_drawing_canvas()

                if image_tensor is not None:
                    # Make prediction
                    with torch.no_grad():
                        image_tensor = image_tensor.to(device)
                        outputs = st.session_state.current_model(image_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        predicted_class = torch.argmax(outputs, dim=1).item()
                        confidence = probabilities[0][predicted_class].item()

                    st.markdown("### Prediction Results")
                    st.markdown(f"**Predicted Digit:** {predicted_class}")
                    st.markdown(f"**Confidence:** {confidence:.2%}")

                    # Show probability distribution
                    prob_values = probabilities[0].cpu().numpy()
                    fig, ax = plt.subplots(figsize=(10, 4))
                    bars = ax.bar(range(10), prob_values)
                    bars[predicted_class].set_color("red")
                    ax.set_xlabel("Digit")
                    ax.set_ylabel("Probability")
                    ax.set_title("Prediction Probabilities")
                    ax.set_xticks(range(10))
                    st.pyplot(fig)

            with col2:
                if image_tensor is not None:
                    st.markdown("### PEEK Maps")

                    # Generate PEEK maps
                    peek_maps, image = generate_peek_maps(image_tensor)

                    if peek_maps:
                        # Create animation-like effect
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for i, (layer, peek_map) in enumerate(peek_maps.items()):
                            status_text.text(
                                f"Generating PEEK map for {layer.__class__.__name__}..."
                            )
                            progress_bar.progress((i + 1) / len(peek_maps))
                            time.sleep(0.5)

                        status_text.text("✅ PEEK maps generated!")
                        progress_bar.progress(1.0)

                        # Plot PEEK maps
                        fig = plot_peek_maps(
                            image,
                            peek_maps,
                            f"PEEK Maps - Predicted: {predicted_class}",
                        )
                        if fig:
                            st.pyplot(fig)
                    else:
                        st.warning(
                            "No PEEK maps generated. Please check if the model has convolutional layers."
                        )

        with tab2:
            st.header("📊 MNIST Test Images")
            st.markdown("Explore PEEK maps for MNIST test images")

            # Load MNIST test dataset
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )

            test_dataset = torchvision.datasets.MNIST(
                root="../data", train=False, transform=transform, download=True
            )

            # Image selection
            col1, col2 = st.columns(2)

            with col1:
                image_index = st.slider(
                    "Select Test Image:", 0, len(test_dataset) - 1, 0
                )
                digit_label = st.selectbox("Or select by digit:", range(10), index=0)

                # Find first occurrence of selected digit
                if st.button("Find First Occurrence"):
                    for i in range(len(test_dataset)):
                        if test_dataset[i][1] == digit_label:
                            image_index = i
                            break

            with col2:
                # Display selected image
                image_tensor, true_label = test_dataset[image_index]
                image_array = image_tensor.squeeze().numpy()

                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(image_array, cmap="gray")
                ax.set_title(f"Test Image {image_index} (True: {true_label})")
                ax.axis("off")
                st.pyplot(fig)

            # Generate PEEK maps
            if st.button("Generate PEEK Maps"):
                with st.spinner("Generating PEEK maps..."):
                    image_tensor = image_tensor.unsqueeze(0).to(device)

                    # Make prediction
                    with torch.no_grad():
                        outputs = st.session_state.current_model(image_tensor)
                        predicted_class = torch.argmax(outputs, dim=1).item()
                        confidence = torch.softmax(outputs, dim=1)[0][
                            predicted_class
                        ].item()

                    # Generate PEEK maps
                    peek_maps, image = generate_peek_maps(image_tensor)

                    if peek_maps:
                        st.markdown(
                            f"**Prediction:** {predicted_class} (Confidence: {confidence:.2%})"
                        )
                        st.markdown(f"**True Label:** {true_label}")

                        # Plot PEEK maps
                        fig = plot_peek_maps(
                            image,
                            peek_maps,
                            f"PEEK Maps - True: {true_label}, Predicted: {predicted_class}",
                        )
                        if fig:
                            st.pyplot(fig)

        with tab3:
            st.header("📈 Model Analysis")
            st.markdown("Analyze model performance and behavior")

            # Model statistics
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Model Architecture")
                st.markdown(f"**Model:** {selected_model}")
                st.markdown(
                    f"**Convolutional Layers:** {len(st.session_state.conv_layers)}"
                )

                for i, layer in enumerate(st.session_state.conv_layers):
                    st.markdown(f"**Layer {i + 1}:** {layer}")

            with col2:
                st.markdown("### Training Information")
                if (
                    "current_model" in st.session_state
                    and st.session_state.current_model is not None
                ):
                    model_path = f"models/{selected_model.lower()}_mnist.pth"
                    checkpoint = torch.load(model_path, map_location=device)

                    st.markdown(f"**Accuracy:** {checkpoint['accuracy']:.2f}%")
                    st.markdown(
                        f"**Training Time:** {checkpoint['training_time']:.2f}s"
                    )
                    st.markdown(f"**Epochs:** {checkpoint['epochs']}")
                    st.markdown(f"**Learning Rate:** {checkpoint['lr']}")

            # Layer analysis
            st.markdown("### Layer Analysis")
            if st.session_state.conv_layers:
                layer_info = []
                for i, layer in enumerate(st.session_state.conv_layers):
                    layer_info.append(
                        {
                            "Layer": f"Conv{i + 1}",
                            "In Channels": layer.in_channels,
                            "Out Channels": layer.out_channels,
                            "Kernel Size": layer.kernel_size[0],
                            "Stride": layer.stride[0],
                            "Padding": layer.padding[0],
                        }
                    )

                st.table(layer_info)

    else:
        st.info("Please load a model from the sidebar to get started!")


if __name__ == "__main__":
    main()
