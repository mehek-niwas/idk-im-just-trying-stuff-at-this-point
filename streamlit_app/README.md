# 🔍 CNN PEEK Map Visualizer

An interactive Streamlit application for visualizing PEEK (Pattern Entropy Extraction and Knowledge) maps from various CNN architectures trained on MNIST.

## Features

- **Multiple CNN Architectures**: Support for SimpleCNN, DeepCNN, VGGNet, and ResNet18
- **Interactive Drawing**: Draw your own digits and see real-time predictions with PEEK maps
- **MNIST Test Images**: Explore PEEK maps for MNIST test dataset images
- **Model Analysis**: Detailed analysis of model architecture and performance
- **Animated PEEK Maps**: Visualize how different convolutional layers focus on different parts of the input

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

First, train all the CNN architectures:

```bash
python train_multiple_models.py
```

This will create trained models in the `models/` directory:
- `simplecnn_mnist.pth`
- `deepcnn_mnist.pth`
- `vggnet_mnist.pth`
- `resnet18_mnist.pth`

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage

### Model Selection
- Use the sidebar to select different CNN architectures
- Click "Load Model" to load the selected model
- View model information including accuracy, training time, and architecture details

### Drawing & Prediction
1. Go to the "🎨 Draw & Predict" tab
2. Draw a digit (0-9) in the canvas
3. See real-time prediction results with confidence scores
4. View PEEK maps showing how each convolutional layer processes your drawing

### MNIST Test Images
1. Go to the "📊 MNIST Test Images" tab
2. Select test images by index or digit label
3. Click "Generate PEEK Maps" to see how the model processes the image
4. Compare true labels with predictions

### Model Analysis
1. Go to the "📈 Model Analysis" tab
2. View detailed information about the model architecture
3. See layer-by-layer analysis of convolutional layers
4. Compare performance metrics across different models

## Model Architectures

### SimpleCNN
- 2 convolutional layers (16 → 32 channels)
- Simple architecture for basic understanding

### DeepCNN
- 3 convolutional layers (32 → 64 → 128 channels)
- Dropout for regularization
- Deeper architecture for better performance

### VGGNet
- VGG-like architecture adapted for MNIST
- 3 blocks with increasing channel depth
- Dense classifier with dropout

### ResNet18
- Residual network with skip connections
- Batch normalization
- Advanced architecture for state-of-the-art performance

## PEEK Maps Explanation

PEEK maps show which regions of the input image each convolutional layer focuses on:

1. **Pattern Detection**: Early layers detect simple patterns (edges, curves)
2. **Feature Combination**: Later layers combine features into more complex patterns
3. **Entropy Visualization**: Brighter regions indicate higher information content
4. **Layer Comparison**: Compare how different layers process the same input

## File Structure

```
streamlit_app/
├── app.py                    # Main Streamlit application
├── train_multiple_models.py  # Training script for all models
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── models/                  # Trained model weights (created after training)
    ├── simplecnn_mnist.pth
    ├── deepcnn_mnist.pth
    ├── vggnet_mnist.pth
    └── resnet18_mnist.pth
```

## Troubleshooting

### No Models Found
If you see "No trained models found", make sure to run `train_multiple_models.py` first.

### CUDA Issues
The application automatically uses GPU if available, otherwise falls back to CPU.

### Memory Issues
For large models (especially VGGNet and ResNet18), ensure you have sufficient GPU memory.

## Performance Tips

- Use GPU for faster inference and PEEK map generation
- Start with SimpleCNN for quick exploration
- Use the drawing canvas for interactive experimentation
- Compare PEEK maps across different model architectures

## Contributing

Feel free to add new model architectures or improve the visualization features! 