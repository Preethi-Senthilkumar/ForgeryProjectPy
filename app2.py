import streamlit as st
from texture import forgery_detection
import os
import uuid

st.title("Forgery Detection App")

# File uploader for uploading the document image
uploaded_file = st.file_uploader("Upload a document image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Create the temp_images directory if it doesn't exist
    if not os.path.exists("temp_images"):
        os.makedirs("temp_images")

    # Save the uploaded file to a temporary location
    temp_image_path = os.path.join("temp_images", str(uuid.uuid4()) + "_" + uploaded_file.name)
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(temp_image_path, caption="Uploaded Image", use_column_width=True)

    # Perform forgery detection when the user clicks the button
    if st.button("Detect Forgery"):
        result = forgery_detection(temp_image_path)
        st.write("Forgery Detection Result:", result)

    # Remove the temporary file
    os.remove(temp_image_path)
