# Cat vs Dog Classifier
This project provides a binary classifier to identify whether an input image is a cat or a dog. It consists of two components:
1. A fine-tuning script (cats_vs_dogs_fine-tuning.py) that trains a ResNet-50 model using a dataset of cat and dog images.
2. A web application (cats_vs_dogs_website.py) built with Streamlit to allow users to upload images and get predictions.

# Prerequisites
1. Python 3.7+
2. Pytorch and Torchvision
3. Streamlit
4. Pillow
5. Google Colab (for replication of fine-tuning)

# Training the Model
1. Organize the dataset into train and test folder, where each folder consists of two subfolders named cats and dogs.
2. Open the fine-tuning.py in Google Colab then make the necessary changes in the base path of the dataset.

# Running the Web Application
1. Download the trained model and make sure to host in GitHub as a release.
2. Run the streamlit app by installing streamlit, and saving website.py in the project folder.
3. Start the app using the command: "streamlit run cats_vs_dogs_website.py"

The dataset used for fine tuning is obtained from Kaggle. You can access the link [here](https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification).
You can access the public website [here](https://cats-vs-dogs-classifier-09199909.streamlit.app/).
