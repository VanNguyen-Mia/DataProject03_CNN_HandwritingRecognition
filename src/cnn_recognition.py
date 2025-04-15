import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeUniform
from sklearn.metrics import confusion_matrix
import numpy as np

# Import the dataset
from datasets.mnist import load_mnist

def build_model():
    """Builds and returns a CNN model."""
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", kernel_initializer=HeUniform(), input_shape=(28,28,1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation="relu", kernel_initializer=HeUniform()),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation="relu", kernel_initializer=HeUniform()),
        Flatten(),
        Dense(64, activation="relu", kernel_initializer=HeUniform()),
        Dense(10, activation="softmax")  # 10 classes for digits 0-9
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss="sparse_categorical_crossentropy", 
                  metrics=["accuracy"])
    return model

def train_model(model, train_images, train_labels, test_images, test_labels, epochs=10, batch_size=64):
    """Trains the CNN model and returns the training history."""
    fit = model.fit(train_images, train_labels, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=(test_images, test_labels))
    return fit

def evaluate_model(model, test_images, test_labels):
    """Evaluates the trained model and returns loss & accuracy."""
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_loss, test_acc

def plot_metrics(fit):
    """Plots training & validation accuracy and loss."""
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(fit.history['accuracy'], label='Training Accuracy')
    plt.plot(fit.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(fit.history['loss'], label='Training Loss')
    plt.plot(fit.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.show()

def plot_confusion_matrix(model, test_images, test_labels):
    """Generates and visualizes the confusion matrix."""
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    cm = confusion_matrix(test_labels, predicted_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def save_model(model, save_path="../saved_model/cnn_model.keras"):
    """Saves the trained model."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved at: {save_path}")