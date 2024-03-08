import streamlit as st
from PIL import Image
import pytesseract
import Levenshtein as lev

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

    # Perform OCR to extract text from the image
    extracted_text = pytesseract.image_to_string(image)

    # Calculate the Levenshtein distance between the extracted text and the original document text
    distance = lev.distance(document_before, extracted_text)

    # Define a threshold for significant changes
    threshold = 50  # Adjust as needed

    # Check if the distance exceeds the threshold
    if distance > threshold:
        st.write(distance)
        st.write("Significant changes detected.")
    else:
        st.write("No significant changes detected.")
