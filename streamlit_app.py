import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import streamlit as st
from PIL import Image
import io
import pandas as pd



class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__() 
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels= 32, kernel_size= 3, stride= 1, padding= 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size= 2)

        self.conv2 = nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size= 3, stride= 1, padding= 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size= 2)


        self.fc1 = nn.Linear(in_features = 64 * 8 * 8, out_features= 128 )
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(in_features = 128, out_features= 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = x.view(-1, 64 * 8 * 8)

        x = self.relu4(self.fc1(x))
        x = self.fc2(x)

        return x

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


@st.cache_resource
def load_model(weights_path: str = 'cifar-10_cnn_model.pth', device: str = 'cpu'):
    model = SimpleCNN()
    state = torch.load(weights_path, map_location= device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def preprocess_image(img: Image.Image) -> torch.Tensor:

    if img.mode != 'RGB':
        img = img.convert('RGB')
    tensor = transform(img)
    return tensor.unsqueeze(0)


def predict(model: nn.Module, tensor: torch.Tensor, device: str = 'cpu'):
    with torch.no_grad():
        tensor = tensor.to(device)
        logits = model(tensor)
        probs = nn.functional.softmax(logits, dim= 1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
    return pred, probs 



st.set_page_config(page_title="CIFAR-10 CNN Demo", page_icon="🧠", layout="centered")
st.title("🧠 CIFAR-10 Object Classifier (PyTorch + Streamlit)")
st.caption("Loads your trained SimpleCNN and predicts uploaded images. "
           "Preprocessing: RGB → 32×32 → Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)).")


device = "cuda" if (torch.cuda.is_available()) else "cpu"
model = load_model("cifar-10_cnn_model.pth", device=device)
st.sidebar.success(f'Model loaded on: {device.upper()}')

mode = st.radio("Choose input mode:", ["Upload an image", "Sample a test digit (from CIFAR-10)"], horizontal=True)

uploaded_img = None
tensor = None
preview_img = None

if mode == "Upload an image":
    file = st.file_uploader("Upload a digit image (PNG/JPG). Tip: clear background, single digit).", 
                            type=["png","jpg","jpeg"])
    if file is not None:
        uploaded_img = Image.open(io.BytesIO(file.read())).convert("RGB")
        preview_img = uploaded_img.copy()
        tensor = preprocess_image(uploaded_img)

else:
    st.info("To begin please pick 'Upload an image'.")
    @st.cache_resource
    def load_cifar_test():
        return datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())
    test_ds = load_cifar_test()
    idx = st.slider("Pick a test index", min_value=0, max_value=len(test_ds)-1, value=0, step=1)
    raw_img, true_label = test_ds[idx]
    pil_img = transforms.ToPILImage()(raw_img)
    preview_img = pil_img.copy()
    tensor = preprocess_image(pil_img)
    st.info(f"True label (for reference): **{true_label}**")

if tensor is not None:
    st.subheader("Input preview")
    st.image(preview_img, caption="Original uploaded/sample image", use_container_width=False, width=200)

    if st.button("🔮 Predict"):
        pred_idx, probs = predict(model, tensor, device)
        pred_name = CIFAR10_CLASSES[pred_idx]
        st.success(f"**Prediction: {pred_idx} — {pred_name}**")

        st.subheader("Class probabilities")
        df = pd.DataFrame({"Objects": CIFAR10_CLASSES, "probability": probs})
        st.bar_chart(df.set_index("Objects"))

        st.subheader("Model input (32×32 after preprocessing)")
        show_img = tensor[0].cpu().permute(1,2,0).numpy()    
        show_img = (show_img * 0.5) + 0.5
        show_img = np.clip(show_img, 0, 1)
        st.image(show_img, caption="What the model actually sees", width=200, clamp=True)

else:
    st.info("Upload an image or pick a test sample to begin.")   