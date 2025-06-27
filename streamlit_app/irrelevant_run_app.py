#!/usr/bin/env python3
"""
Launcher script for the CNN PEEK Map Visualizer
"""

import os
import sys
import subprocess


def check_models():
    """Check if trained models exist"""
    models_dir = "models"
    required_models = [
        "simplecnn_mnist.pth",
        "deepcnn_mnist.pth",
        "vggnet_mnist.pth",
        "resnet18_mnist.pth",
    ]

    if not os.path.exists(models_dir):
        return False, "Models directory not found"

    missing_models = []
    for model in required_models:
        if not os.path.exists(os.path.join(models_dir, model)):
            missing_models.append(model)

    if missing_models:
        return False, f"Missing models: {', '.join(missing_models)}"

    return True, "All models found"


def main():
    print("🔍 CNN PEEK Map Visualizer Launcher")
    print("=" * 50)

    # Check if models exist
    models_ok, message = check_models()

    if not models_ok:
        print(f"❌ {message}")
        print("\n📋 To train the models, run:")
        print("   python train_multiple_models.py")
        print("\n⏱️  This will take some time to train all models.")
        print("   You can also train individual models by modifying the script.")
        return

    print("✅ All trained models found!")
    print("\n🚀 Starting Streamlit application...")
    print("   The app will open in your browser at http://localhost:8501")
    print("   Press Ctrl+C to stop the application")
    print("\n" + "=" * 50)

    # Run Streamlit
    try:
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        print("\n💡 Make sure Streamlit is installed:")
        print("   pip install streamlit")
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except FileNotFoundError:
        print("❌ Streamlit not found!")
        print("\n💡 Install Streamlit with:")
        print("   pip install streamlit")


if __name__ == "__main__":
    main()
