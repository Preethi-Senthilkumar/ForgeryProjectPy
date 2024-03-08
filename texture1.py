import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pickle

def gabor_filter(img, ksize=9, sigma=3.0, theta=0, lam=1.0, gamma=0.02, psi=0, ktype=cv2.CV_32F):
    """
    Apply Gabor filter to the input image.
    """
    kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, gamma, psi, ktype)
    kern /= 1.5*kern.sum()
    return cv2.filter2D(img, cv2.CV_8UC3, kern)

def extract_features(img):
    """
    Extract features from the preprocessed image.
    """
    # Check if the image is grayscale (2 dimensions) or color (3 dimensions)
    if len(img.shape) > 2:  # Color image
        num_channels = img.shape[2]
    else:  # Grayscale image
        num_channels = 1
    
    # Compute the mean, standard deviation, min, and max pixel values for each color channel
    features = []
    for channel in range(num_channels):
        if num_channels > 1:
            channel_values = img[:, :, channel]
        else:
            channel_values = img

        mean_value = np.mean(channel_values)
        std_value = np.std(channel_values)
        min_value = np.min(channel_values)
        max_value = np.max(channel_values)
        features.extend([mean_value, std_value, min_value, max_value])

    # Add more features if needed to ensure there are 10 features
    while len(features) < 10:
        features.append(0.0)  # Placeholder value

    # Return the extracted features as a numpy array
    return np.array(features)


# Load the trained model
with open('model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

st.title('Document Forgery Detection')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    image_np = np.array(image)

    # Check if the image is not already in grayscale
    if len(image_np.shape) > 2 and image_np.shape[2] > 1:
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image_np

    filtered_image = gabor_filter(gray_image)

    # Display the preprocessed image
    st.image(filtered_image, caption='Filtered Image', use_column_width=True)

    # Extract features from the preprocessed image
    features = extract_features(filtered_image)

    # Make a prediction using the loaded model
    prediction = clf.predict(features.reshape(1, -1))

    if prediction == 0:
        st.write("The image is genuine.")
    else:
        st.write("The image is forged.")
