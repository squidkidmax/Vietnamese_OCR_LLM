import easyocr

reader = easyocr.Reader(['vi'])  # 'vi' = Vietnamese
results = reader.readtext('handwriting_image.jpg')

for bbox, text, confidence in results:
    print(f"{text} (confidence: {confidence:.2f})")
