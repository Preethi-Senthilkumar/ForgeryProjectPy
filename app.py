

import streamlit as st
import cv2
import numpy as np
import pytesseract

# Function to extract text from an image
def extract_text(image_path):
    # Read the image
    image = cv2.imread(image_path)

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

     # Function to detect graphics using an object detection model (e.g., YOLO)
def detect_graphics(image_path):
    # Load pre-trained object detection model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


    # Read the image
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    # Detect objects
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    graphics = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
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
            graphic_center = (graphic_element[0] + graphic_element[2] // 2, graphic_element[1] + graphic_element[3] // 2)
            distance = euclidean_distance(text_center, graphic_center)
            distances.append(distance)

    # Check if any distance is less than the threshold
    if any(distance < threshold for distance in distances):
        return "Aligned"
    else:
        return "Forgery"
    
    def detect_text(image_path):
    # Load the pre-trained EAST text detector
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    # Read the image
    image = cv2.imread(image_path)
    orig = image.copy()
    (H, W) = image.shape[:2]

    # Preprocess the image for EAST
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)

    # Forward pass through the EAST network
    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    # Decode the predictions
    rects, confidences = decode_predictions(scores, geometry)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # Iterate over the bounding boxes and draw them on the image
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    return orig

# Main function
def main(uploaded_file):
    # Extract text from the uploaded image
    text_elements = extract_text(uploaded_file)

    # Detect graphics in the uploaded image
    graphics = detect_graphics(uploaded_file)

    # Analyze alignment
    alignment_result = analyze_alignment(text_elements, graphics)

    return alignment_result





# Streamlit UI
st.title("Document Forgery Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    #st.image(image, caption="Uploaded Image", use_column_width=True)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)


    # Extract text from the uploaded image
    #text_elements = extract_text(uploaded_file)

    import os
    import tempfile


def extract_text(uploaded_file):
    # Save the uploaded file to a temporary location
    temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Read the image from the temporary file
    image = cv2.imread(temp_file_path)

    # Continue with your processing...

    # Remove the temporary file
    os.remove(temp_file_path)

    return text_elements


    # Detect graphics in the uploaded image
    graphics = detect_graphics(uploaded_file)

    # Analyze alignment
    alignment_result = analyze_alignment(text_elements, graphics)

    # Display the result
    st.write("Alignment Result:", alignment_result)
