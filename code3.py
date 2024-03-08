import streamlit as st
import numpy as np
import cv2
import pytesseract
import tempfile
import os
from PIL import Image

# ... (previous code unchanged) ...

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to extract text from an image
def extract_text(uploaded_file):
    # Read the image
    image = Image.open(uploaded_file)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use pytesseract to perform OCR
    results = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    # Extract bounding box coordinates of each text element
    text_elements = []
    for i in range(len(results["text"])):
        if int(results["conf"][i]) > 0:  # Filter out low-confidence detections
            x = int(results["left"][i])
            y = int(results["top"][i])
            w = int(results["width"][i])
            h = int(results["height"][i])
            text_elements.append((x, y, w, h))

    return text_elements


import cv2

def detect_graphics(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Perform your graphics detection logic here

    # For example, you can use a simple thresholding technique
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the image (for visualization)
    result = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

    # Check if contours were found (graphics detected)
    is_tampered = len(contours) > 0

    return result, is_tampered

def main(uploaded_file):
    # ... (previous code unchanged) ...
      temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
      with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Detect graphics in the uploaded image
      graphics_image, is_tampered = detect_graphics(temp_file_path)

    # Remove the temporary file
      os.remove(temp_file_path)

      return graphics_image, is_tampered

    # Analyze the uploaded image and predict tampering
      is_tampered = predict_tampering(graphics_image)  # Call the prediction function

      return graphics_image, is_tampered

# Streamlit UI
st.title("Document Forgery Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # ... (previous code unchanged) ...
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Analyze the uploaded image
    result, is_tampered = main(uploaded_file)

    

    # Display the result
    result_image, is_tampered = main(uploaded_file)
    st.image(result_image, caption="Detected Graphics", use_column_width=True)

    st.write("Prediction:")
    if is_tampered:
        st.write("The image is likely tampered.")
    else:
        st.write("The image appears to be genuine.")
