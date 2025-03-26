from src.cnn_recognition import *
from datasets.mnist import load_mnist

if __name__ == "__main__":
    # Load data
    train_images, train_labels, test_images, test_labels = load_mnist()

    # Build, train, evaluate, and save model
    cnn_model = build_model()
    history = train_model(cnn_model, train_images, train_labels, test_images, test_labels)
    evaluate_model(cnn_model, test_images, test_labels)

    # Visualizations
    plot_metrics(history)
    plot_confusion_matrix(cnn_model, test_images, test_labels)

    # Save model
    save_model(cnn_model)