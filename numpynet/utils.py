import numpy as np
import pickle

def to_categorical(labels):
    categories = np.max(labels) + 1
    cat_labels = np.zeros((len(labels), categories))
    for i, label in enumerate(labels):
        cat_labels[i, label] = 1
    return cat_labels

def load_mnist(path: str):
    with open(path, "rb") as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]
