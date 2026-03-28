import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("model/pneumonia_model.h5")
classes = ["BACTERIA", "NORMAL", "VIRUS"]

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.reshape(img, (1,224,224,3))

    pred = model.predict(img)
    class_index = np.argmax(pred)
    confidence = np.max(pred)

    return classes[class_index], confidence