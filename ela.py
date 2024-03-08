import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import tempfile
import os

# Function to compute ELA
def compute_ela(path, quality):
    temp_filename = 'temp_file_name.jpg'
    SCALE = 15
    orig_img = cv2.imread(path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(temp_filename, orig_img, [cv2.IMWRITE_JPEG_QUALITY, quality])

    # Read the compressed image
    compressed_img = cv2.imread(temp_filename)

    # Get the absolute difference between original and compressed images and multiply by scale
    diff = SCALE * cv2.absdiff(orig_img, compressed_img)
    return diff

# Function to convert image to ELA image
def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image

# Streamlit UI
st.title("Document Forgery Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(temp_file_path, caption="Uploaded Image", use_column_width=True)

    # Compute ELA
    quality = 90  # JPEG compression quality
    ela_diff = compute_ela(temp_file_path, quality)

    # Display the ELA difference image
    st.image(ela_diff, caption="ELA Difference", use_column_width=True)

    # Remove the temporary file
    os.remove(temp_file_path)
