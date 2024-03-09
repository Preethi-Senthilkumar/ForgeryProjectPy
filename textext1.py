import cv2
import pytesseract
import streamlit as st
import numpy as np
from PIL import Image

# Download the pre-trained OCR model from Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_regions(image):
    # Convert the image to a NumPy array
    image_np = np.array(image)

    # Check if the image is already in grayscale
    if len(image_np.shape) == 2:
        gray_image = image_np
    else:
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Apply thresholding to get a binary image
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask image to store the text regions
    mask = np.zeros_like(gray_image)

    # Iterate through the contours and draw them on the mask image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        area = cv2.contourArea(contour)
        if aspect_ratio < 5 and area > 1000:  # Filter out non-text contours based on aspect ratio and area
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    # Apply the mask to the original image to extract the text regions
    text_regions = cv2.bitwise_and(image_np, image_np, mask=mask)

    return text_regions


def detect_text_changes(image):
    # Extract text regions from the image
    text_regions = extract_text_regions(image)

    # Convert the text regions to grayscale
    text_regions_gray = cv2.cvtColor(text_regions, cv2.COLOR_RGB2GRAY)

    # Use Tesseract OCR to extract text from the text regions
    text = pytesseract.image_to_string(text_regions_gray)

    # Check if the text contains any characters that indicate overwriting
    if any(char.isalpha() for char in text):
        st.write("Result: This document may contain overwritten text (Forged)")
    else:
        st.write("Result: This document does not contain overwritten text (Genuine)")

    # Display the extracted text
    st.write("Extracted Text:")
    st.write(text)

def main():
    st.title("Text Forgery Detection")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        detect_text_changes(image)

if __name__ == "__main__":
    main()
