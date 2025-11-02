🧠 CIFAR-10 Image Classifier (Streamlit App)

A simple Streamlit application that loads a pre-trained SimpleCNN model and predicts the class of an image from the CIFAR-10 dataset.

## How to Use

1. Choose a mode:
   - Upload an image — upload your own image.
   - Sample a test digit (from CIFAR-10) — select an image by index from the CIFAR-10 test set.
2. Click Predict.
3. You will see:
   - The predicted index and class name (from `CIFAR10_CLASSES`).
   - A probability histogram across all 10 classes.
   - The input image (32×32, normalized) that was fed into the model.

## Requirements

Before running the app:
1) Install the required libraries:
pip install -r requirements.txt

2) Make sure the model weights are in the same directory as streamlit_app.py:
cifar-10_cnn_model.pth

3) Run the app in your terminal:
streamlit run streamlit_app.py

Once started, a browser window will open with an interface where you can upload or select an image and get the model’s prediction.