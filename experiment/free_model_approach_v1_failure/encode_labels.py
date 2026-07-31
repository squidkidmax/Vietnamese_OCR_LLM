import numpy as np

CHAR_LIST = "()+,.-/0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹẲẴẶẺẼỈỊỎỖỘỬỮ;'\"%&@#*!?"
NUM_CLASSES = len(CHAR_LIST) + 1  # 138

def encode_labels(text_labels):
    char_to_num = {char: idx + 1 for idx, char in enumerate(CHAR_LIST)}  # 1-based indices (1 to 137)
    blank_label = NUM_CLASSES - 1  # 137 for blank
    
    encoded_labels = []
    for idx, label in enumerate(text_labels):
        # Skip unknown characters
        encoded = [char_to_num[c] for c in label if c in char_to_num]
        if not encoded:  # If no valid chars, use blank
            encoded = [blank_label]
        # Debug: Check for invalid indices
        if any(i >= NUM_CLASSES for i in encoded):
            print(f"Invalid label at index {idx}: {label}, encoded: {encoded}")
        encoded_labels.append(encoded)
    
    max_label_len = max(len(l) for l in encoded_labels)
    
    # Pad labels with -1 for CTC
    padded_labels = np.ones((len(encoded_labels), max_label_len)) * -1
    for i, label in enumerate(encoded_labels):
        padded_labels[i, :len(label)] = label
    
    label_lengths = np.array([len(label) for label in encoded_labels])
    
    return padded_labels, label_lengths, max_label_len