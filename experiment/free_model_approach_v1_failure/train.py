import numpy as np
from sklearn.model_selection import train_test_split
from preprocess import preprocess_images
from encode_labels import encode_labels, CHAR_LIST, NUM_CLASSES
from build_model import build_crnn_model

# Parameters
IMG_HEIGHT = 128
MAX_WIDTH = 2048
BATCH_SIZE = 16
EPOCHS = 50

def train_model(images, text_labels):
    # Preprocess
    processed_images = preprocess_images(images)
    
    # Encode
    padded_labels, label_lengths, max_label_len = encode_labels(text_labels)
    
    # Debug: Inspect labels
    print(f"NUM_CLASSES: {NUM_CLASSES}, CHAR_LIST length: {len(CHAR_LIST)}")
    print("First 16 labels:")
    for i in range(min(16, len(text_labels))):
        valid_labels = padded_labels[i][:label_lengths[i]]
        print(f"Index {i}: Original = {text_labels[i]}, Encoded = {valid_labels}")
        if any(l >= NUM_CLASSES for l in valid_labels):
            print(f"Invalid index at {i}: {valid_labels}")
    
    # Input lengths (4x downsample)
    input_lengths = np.full((len(processed_images),), MAX_WIDTH // 4)
    
    # Split
    train_images, val_images, train_labels, val_labels, train_input_len, val_input_len, train_label_len, val_label_len = train_test_split(
        processed_images, padded_labels, input_lengths, label_lengths, test_size=0.2
    )
    
    # Build model
    model = build_crnn_model(IMG_HEIGHT, MAX_WIDTH, NUM_CLASSES, max_label_len)
    model.compile(optimizer='adam', loss={'ctc': lambda y_true, y_pred: y_pred})
    
    # Train
    model.fit(
        x={'image': train_images, 'labels': train_labels, 'input_length': train_input_len, 'label_length': train_label_len},
        y=np.zeros(len(train_images)),
        validation_data=({'image': val_images, 'labels': val_labels, 'input_length': val_input_len, 'label_length': val_label_len}, np.zeros(len(val_images))),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2
    )
    
    # Save
    model.save('crnn_model_trained.h5')

if __name__ == "__main__":
    from load_data import load_data
    images, text_labels = load_data()
    train_model(images, text_labels)