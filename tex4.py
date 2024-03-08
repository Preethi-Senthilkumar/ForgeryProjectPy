import streamlit as st
from PIL import Image
import cv2
import numpy as np
import Levenshtein as lev
import pytesseract
import os
import tempfile

# Load Tesseract OCR engine
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Sample document content before editing
document_before = "This is the original document content."

# Function to extract text from an image
import os
import tempfile

def extract_text(uploaded_file):
    # Save the uploaded file to a temporary location
    temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Read the image
    image = cv2.imread(temp_file_path)

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

    # Remove the temporary file
    os.remove(temp_file_path)

    return text_elements


# Function to detect graphics using an object detection model (e.g., YOLO)
def detect_graphics(uploaded_file):
    # Read the image
    image = cv2.imread(uploaded_file)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get a binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process contours to get bounding boxes
    graphics = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        graphics.append((x, y, w, h))

    return graphics


# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to analyze alignment between text and graphics
def analyze_alignment(text, graphics, threshold=50):
    # For each text and graphic element, calculate the distance between them
    distances = []
    for text_element in text:
        for graphic_element in graphics:
            text_center = (text_element[0] + text_element[2] // 2, text_element[1] + text_element[3] // 2)
            graphic_center = (graphic_element[0], graphic_element[1])
            distance = euclidean_distance(text_center, graphic_center)
            distances.append(distance)

    # Check if any distance is less than the threshold
    if any(distance < threshold for distance in distances):
        return "Aligned"
    else:
        return "Forgery"

# Main Streamlit app
def main():
    # Upload an image of the document
    uploaded_image = st.file_uploader("Upload an image of the document", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Read the uploaded image
        image = Image.open(uploaded_image)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform OCR to extract text from the image
        text_elements = extract_text(uploaded_image)

        # Detect graphics in the image
        graphics = detect_graphics(uploaded_image)

        # Analyze alignment between text and graphics
        alignment_result = analyze_alignment(text_elements, graphics)

        # Display the alignment result
        st.write(f"Alignment result: {alignment_result}")

if __name__ == "__main__":
    main()
