# 6. evaluate.py
# This script evaluates the trained model on validation data or new data.

import tensorflow as tf
import numpy as np
from preprocess import preprocess_images
from encode_labels import encode_labels

CHAR_LIST = '()+,-./0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstuvwxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ'  # Same charset as in infer.py
# Assume preprocess_images, encode_labels

def evaluate_model(model_path, test_images, test_text_labels):
    model = tf.keras.models.load_model(model_path, compile=False)  # Load without compiling for prediction
    
    # Preprocess test images
    processed_test = preprocess_images(test_images)
    
    # For prediction, we need the prediction model (without CTC loss inputs)
    prediction_model = tf.keras.models.Model(
        model.get_layer(name='image').input,
        model.get_layer(name='predictions').output
    )
    
    # Predict
    preds = prediction_model.predict(processed_test)
    
    # Decode using CTC
    decoded_preds = tf.keras.backend.ctc_decode(preds, input_length=np.full((len(processed_test),), preds.shape[1]), greedy=True)[0][0]
    decoded_texts = []
    for pred in decoded_preds:
        text = ''.join([CHAR_LIST[int(p)] for p in pred if int(p) != -1])
        decoded_texts.append(text)
    
    # Compare with ground truth
    correct = sum(1 for pred, true in zip(decoded_texts, test_text_labels) if pred == true)
    accuracy = correct / len(test_text_labels)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Example usage: evaluate_model('crnn_model_trained.h5', test_images, test_text_labels)