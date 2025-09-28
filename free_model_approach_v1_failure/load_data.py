import os
import json
import cv2
import numpy as np

DATA_DIR = './vn_handwritten_images/data'
LABELS_FILE = './vn_handwritten_images/labels.json'

def load_data():
    with open(LABELS_FILE, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    images = []
    text_labels = []
    for filename, label in labels.items():
        img_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                text_labels.append(label)
    
    return images, text_labels