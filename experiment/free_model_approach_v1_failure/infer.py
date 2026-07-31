# 7. infer.py
# This script performs inference on a new image using the trained model.

import tensorflow as tf
import cv2
import numpy as np
from build_model import build_crnn_model
from encode_labels import CHAR_LIST, NUM_CLASSES  # Ensure consistency
from preprocess import preprocess_images


CHAR_LIST = '()+,-./0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstuvwxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ'

def infer_on_image(model_path, image_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    prediction_model = tf.keras.models.Model(
        model.get_layer(name='image').input,
        model.get_layer(name='predictions').output
    )
    
    # Load and preprocess single image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Image not found"
    
    # Use preprocess function (adapt for single image)
    processed_img = preprocess_images([img])[0]
    processed_img = np.expand_dims(processed_img, axis=0)  # Batch dim
    
    # Predict
    pred = prediction_model.predict(processed_img)
    decoded = tf.keras.backend.ctc_decode(pred, input_length=np.full((1,), pred.shape[1]), greedy=True)[0][0]
    text = ''.join([CHAR_LIST[int(p)] for p in decoded[0] if int(p) != -1])
    
    return text

# Example: result = infer_on_image('crnn_model_trained.h5', 'path/to/new/image.png')
# print(result)