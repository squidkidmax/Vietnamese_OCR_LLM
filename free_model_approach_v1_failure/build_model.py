import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from encode_labels import NUM_CLASSES, CHAR_LIST  # Ensure consistency

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def build_crnn_model(img_height, max_width, num_classes, max_label_len):
    input_img = layers.Input(shape=(img_height, max_width, 1), name='image')
    
    # CNN Layers
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Conv2D(512, (2, 2), padding='valid', activation='relu')(x)
    
    # Reshape for RNN
    x = layers.Reshape(target_shape=(-1, 512))(x)
    
    # RNN
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    
    # Output
    y_pred = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    # CTC inputs
    labels_input = layers.Input(name='labels', shape=[max_label_len])
    input_length = layers.Input(name='input_length', shape=[1])
    label_length = layers.Input(name='label_length', shape=[1])
    
    ctc_loss = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels_input, input_length, label_length])
    
    model = models.Model(inputs=[input_img, labels_input, input_length, label_length], outputs=ctc_loss)
    return model