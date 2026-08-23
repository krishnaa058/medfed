"""
MedFed — Live Brain Tumor MRI Classifier
"""

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

st.set_page_config(page_title="MedFed — Live Prediction", page_icon="🧠", layout="centered")

st.title("🧠 MedFed — Live Brain Tumor MRI Classifier")
st.caption("This model was trained using Privacy-Preserving Federated Learning across 3 simulated hospitals — no patient image ever left its institution during training.")

@st.cache_resource
def load_model():
    model = torch.jit.load('medfed_model.pt', map_location='cpu')
    model.eval()
    return model

model = load_model()

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Upload a brain MRI scan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded MRI scan", width=300)

    with st.spinner("Running inference on the federated model..."):
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]

    st.subheader("Prediction")
    results = {classes[i]: float(probs[i]) for i in range(len(classes))}
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    top_class = list(sorted_results.keys())[0]
    top_conf = list(sorted_results.values())[0]
    st.success(f"**{top_class.upper()}** — {top_conf*100:.1f}% confidence")

    for cls, prob in sorted_results.items():
        st.write(f"{cls}")
        st.progress(prob)

else:
    st.info("👆 Upload an MRI image to see a live prediction from the federated model.")

st.divider()
st.caption("Model trained via Federated Averaging (FedAvg) across Hospital_1, Hospital_2, Hospital_3 · Part of the MedFed platform · H-03 Hackathon Project")
