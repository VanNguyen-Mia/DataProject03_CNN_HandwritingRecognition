import numpy as np
import tensorflow as tf
from PIL import Image

def load_trained_model(model_path):
    """Loads a trained CNN model from the specified path."""
    return tf.keras.models.load_model(model_path)

def preprocess_image(image):
    """Preprocesses an image (uploaded or drawn) for CNN model inference."""
    image = image.resize((28, 28)).convert("L")  # Convert to 28x28 grayscale
    image_array = np.array(image) / 255.0  # Normalize to [0,1]
    image_array = np.expand_dims(image_array, axis=(0, -1))  # Reshape for CNN (1,28,28,1)
    return image_array

def predict_digit(model, image_array):
    """Predicts the digit from the preprocessed image."""
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)
    confidence = prediction[0][predicted_digit] * 100
    return predicted_digit, prediction, confidence