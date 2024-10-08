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
GOOGLE_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSfQbGpZgcCbWNlI2mdAxmGq8gXae0jZx88haKqTQsmAj_uzDA/viewform"

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

# Function to upload an image to Google Drive
def upload_to_gdrive(image, filename):
    # Authenticate and create the PyDrive client
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    
    # Save image temporarily
    temp_path = f"/tmp/{filename}"
    image.save(temp_path)
    
    # Create and upload file
    gfile = drive.CreateFile({'title': filename})
    gfile.SetContentFile(temp_path)
    gfile.Upload()
    
    st.success("Image uploaded to Google Drive.")

# Streamlit application
def main():
    st.title('Cat vs Dog Classifier')
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        model = load_model()
        probability_cat, probability_dog = predict_image(image, model)
        
        st.image(image.resize((300, 300)), caption='Successfully Uploaded Image', use_column_width=True)
        
        if probability_dog > probability_cat:
            st.markdown(f"<h2 style='color: red;'>Aha! I'm {probability_dog:.4f}% that it is a Dog</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color: blue;'>Aha! I'm {probability_cat:.4f}% that it is a Cat</h2>", unsafe_allow_html=True)
        
        # Ask the user if the prediction is correct
        st.write("Is the prediction correct?")
        if st.button("Yes"):
            st.write("Great! Thanks for confirming.")
        elif st.button("No"):
            st.write("Oh no! Please submit the correct image using the form below:")
            st.markdown(f"[Submit Image via Google Form]({GOOGLE_FORM_URL})")

if __name__ == "__main__":
    main()
