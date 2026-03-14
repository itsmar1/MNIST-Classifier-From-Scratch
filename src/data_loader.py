import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


np.random.seed(42)

class DataLoader:
    @staticmethod
    def load_mnist():
        print("Loading MNIST Data...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist.data, mnist.target.astype(int)
        X = X.astype(np.float32) / 255.0
        return X, y

    @staticmethod
    def preprocess_data(X, y, validation_split=0.2, test_split=0.1):
        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_split, random_state=42)

        # Second split: train vs validation
        val_relative_size = validation_split / (1- test_split)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_relative_size, random_state=42)

        # One-hot encode labels
        encoder = OneHotEncoder(sparse_output=False)
        y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
        y_val_encoded = encoder.transform(y_val.reshape(-1, 1))
        y_test_encoded = encoder.transform(y_test.reshape(-1,1))

        return (X_train, X_val, X_test,
                y_train_encoded, y_val_encoded, y_test_encoded,
                y_train, y_val, y_test)


    @staticmethod
    def create_batches(X, y, batch_size):
        """Create mini-batches"""
        m_samples = X.shape[0]
        indices = np.random.permutation(m_samples)
        for i in range(0, m_samples, batch_size):
            batch_indices = indices[i : i + batch_size]
            yield X[batch_indices], y[batch_indices]


