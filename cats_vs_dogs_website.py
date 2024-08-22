import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Define the same preprocessing steps as in training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Lambda(lambda img: img.convert("RGB")),  # Ensure 3 channels
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
        nn.Linear(num_features, 1),  # Output for binary classification
        nn.Sigmoid()  # Sigmoid for probability
    )
    
    # Load the saved model weights
    model.load_state_dict(torch.load('cats_vs_dogs_model.pth', map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    
    return model

# Predict function
def predict_image(image, model):
    img = transform(image)  # Apply transformations
    img = img.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(img)
    
    probability_dog = output.item()
    probability_cat = 1 - probability_dog
    
    return probability_cat, probability_dog

# Streamlit application
def main():
    st.title('Cat vs Dog Classifier')
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        
        # Load the model
        model = load_model()
        
        # Predict
        probability_cat, probability_dog = predict_image(image, model)
        
        # Display results
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write()  # Add an empty line for spacing
        st.write(f"Probability of being a Cat: {probability_cat:.4f}")
        st.write(f"Probability of being a Dog: {probability_dog:.4f}")
        
        # Determine and display the result with larger font
        if probability_dog > probability_cat:
            st.markdown(f"<h2 style='color: red;'>The image is predicted to be a Dog with probability {probability_dog:.4f}.</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color: blue;'>The image is predicted to be a Cat with probability {probability_cat:.4f}.</h2>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
