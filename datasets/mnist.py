import tensorflow as tf
from tensorflow.keras import datasets

# Load MNIST dataset
def load_mnist():
    """Load and preprocess the MNIST dataset."""
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))  # (60000, 28, 28, 1)
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))  # (10000, 28, 28, 1)
    # Normalize pixel values to [0, 1]
    train_images, test_images = train_images / 255.0, test_images / 255.0
    
    return train_images, train_labels, test_images, test_labels

