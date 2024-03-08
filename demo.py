import streamlit as st
 
# st.title('My First Streamlit App')
import cv2
import numpy as np
import pytesseract
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
import string
import nltk
import os
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.parse import CoreNLPParser
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tree import ParentedTree
from transformers import AutoTokenizer, AutoModel
import torch
from nltk.wsd import lesk
from collections import defaultdict



nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')


# Function to perform text extraction using OCR
def extract_text(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use pytesseract to perform OCR
    text = pytesseract.image_to_string(gray)

    return text

# Function to detect graphics using an object detection model (e.g., YOLO)
def detect_graphics(image_path):
    # Load pre-trained object detection model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

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

# Main function
def main(image_path):
    # Extract text from the image
    text = extract_text(image_path)

    # Detect graphics in the image
    graphics = detect_graphics(image_path)

    # Analyze alignment
    alignment_result = analyze_alignment(text, graphics)

    return alignment_result

# Example usage
image_path = "document_image.jpg"
result = main(image_path)
print("Alignment Result:", result)
