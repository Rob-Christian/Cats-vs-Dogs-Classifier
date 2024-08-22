import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import requests
import os

# Define the URL for the model file hosted on GitHub Releases
MODEL_URL = 'https://github.com/Rob-Christian/Cats-vs-Dogs-Classifier/releases/download/v1.0.0/cats_vs_dogs_model.pth'
MODEL_PATH = 'cats_vs_dogs_model.pth'

# Function to download the model file from GitHub Releases
@st.cache_data
def download_model(url, path):
    if not os.path.exists(path):
        response = requests.get(url)
        response.raise_for_status()  # Ensure we notice bad responses
        with open(path, 'wb') as f:
            f.write(response.content)

# Define the preprocessing steps
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
])

# Load the model
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=True)
    
    # Freeze convolutional layers
    for param in model.parameters():
        param.requires_grad = False

    # Modify the fully connected layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1),
        nn.Sigmoid()
    )
    
    # Download the model if necessary
    download_model(MODEL_URL, MODEL_PATH)
    
    # Load the model weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    
    return model

# Prediction function
def predict_image(image, model):
    img = transform(image)
    img = img.unsqueeze(0)

    with torch.no_grad():
        output = model(img)
    
    probability_dog = 100 * output.item()
    probability_cat = 100 - probability_dog
    
    return probability_cat, probability_dog

# Streamlit application
def main():
    st.title('Cat vs Dog Classifier')
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        model = load_model()
        probability_cat, probability_dog = predict_image(image, model)
        
        st.image(image, caption='Successfully Uploaded Image', use_column_width=True)
        
        if probability_dog > probability_cat:
            st.markdown(f"<h2 style='color: red;'>Aha! It is a Dog (confidence level: {probability_dog:.4f}%)</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color: blue;'>Aha! It is a Cat (confidence level: {probability_cat:.4f}%)</h2>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
