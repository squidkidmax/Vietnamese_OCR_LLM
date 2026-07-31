# 2. preprocess.py
# This script preprocesses the images: resize to fixed height, pad to max width, normalize.
# Adjusted for larger book-like images.

import cv2
import numpy as np

MAX_WIDTH = 2048  # Adjustable for larger images
IMG_HEIGHT = 128  # Adjustable for taller images

def preprocess_images(images):
    processed_images = []
    for img in images:
        # Resize to fixed height, maintain aspect ratio
        ratio = IMG_HEIGHT / img.shape[0]
        new_width = int(img.shape[1] * ratio)
        img_resized = cv2.resize(img, (new_width, IMG_HEIGHT))
        
        # Pad to max width
        if new_width < MAX_WIDTH:
            pad = np.ones((IMG_HEIGHT, MAX_WIDTH - new_width)) * 255
            img_padded = np.hstack((img_resized, pad))
        elif new_width > MAX_WIDTH:
            img_padded = cv2.resize(img_resized, (MAX_WIDTH, IMG_HEIGHT))
        else:
            img_padded = img_resized
        
        # Normalize
        img_padded = img_padded / 255.0
        img_padded = np.expand_dims(img_padded, axis=-1)  # Add channel
        
        processed_images.append(img_padded)
    
    return np.array(processed_images)

# Example usage (assuming images loaded from load_data.py)
# preprocessed = preprocess_images(images)