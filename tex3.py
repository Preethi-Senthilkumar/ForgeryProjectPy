import streamlit as st
from PIL import Image
import cv2
import numpy as np
import Levenshtein as lev
import pytesseract

# Load Tesseract OCR engine
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Sample document content before editing
document_before = "This is the original document content."

# Upload an image of the document
uploaded_image = st.file_uploader("Upload an image of the document", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read the uploaded image
    image = Image.open(uploaded_image)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Perform OCR to extract text from the image
    extracted_text = pytesseract.image_to_string(gray_image)

    # Calculate the Levenshtein distance between the extracted text and the original document text
    distance = lev.distance(document_before, extracted_text)

    # Define a threshold for significant changes
    text_threshold = 5  # Adjust as needed
    geometric_threshold = 0.8  # Adjust as needed

    # Perform geometric analysis
    contours, _ = cv2.findContours(cv2.Canny(gray_image, 100, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        bounding_box_area_ratio = cv2.contourArea(contours[0]) / (gray_image.shape[0] * gray_image.shape[1])
    else:
        bounding_box_area_ratio = 0.0

    # Check if the distance exceeds the text threshold or bounding box area ratio is below the geometric threshold
    if distance > text_threshold or bounding_box_area_ratio < geometric_threshold:
        st.write("Significant changes detected.")
    else:
        st.write("No significant changes detected.")
