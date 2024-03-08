import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def texture_analysis(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute LBP image
    radius = 3
    num_points = 8 * radius
    lbp = local_binary_pattern(gray, num_points, radius, method='uniform')
    lbp = np.uint8(lbp * 255)

    return lbp

def calculate_inconsistency(lbp_image):
    # Calculate texture inconsistency using variance of the LBP image
    inconsistency = np.var(lbp_image)

    return inconsistency

def forgery_detection(document_image_path, threshold=1500):
    lbp_image = texture_analysis(document_image_path)
    document_texture_inconsistency = calculate_inconsistency(lbp_image)
    if document_texture_inconsistency > threshold:
        return "forged"
    else:
        return "genuine"
