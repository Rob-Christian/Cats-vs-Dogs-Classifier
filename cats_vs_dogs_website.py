import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import requests
import os

# Define the URL for the model file
MODEL_URL = 'https://drive.google.com/uc?export=download&id=1wqvISJDJ-Q3ZDpssZge4L6DRXM-3ElsY'
MODEL_PATH = '/content/drive/My Drive/My Career/Projects/cats and dogs/cats_vs_dogs_model.pth'

# Function to download the model file
def download_model(url, path):
    if not os.path.exists(path):
        response = requests.get(url)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                f.write(response.content)
            st.write("Model downloaded successfully.")
        else:
            st.error(f"Failed to download model file. Status code: {response.status_code}")
    else:
        st.write("Model file already exists. Using existing file.")

# Define the preprocessing steps
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
])

# Load the model
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
    
    probability_dog = output.item()
    probability_cat = 1 - probability_dog
    
    return probability_cat, probability_dog

# Streamlit application
def main():
    st.title('Cat vs Dog Classifier')
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        model = load_model()
        probability_cat, probability_dog = predict_image(image, model)
        
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write()
        st.write(f"Probability of being a Cat: {probability_cat:.4f}")
        st.write(f"Probability of being a Dog: {probability_dog:.4f}")
        
        if probability_dog > probability_cat:
            st.markdown(f"<h2 style='color: red;'>The image is predicted to be a Dog with probability {probability_dog:.4f}.</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color: blue;'>The image is predicted to be a Cat with probability {probability_cat:.4f}.</h2>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
