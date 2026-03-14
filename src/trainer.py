import numpy as np
from tqdm import tqdm

from src.data_loader import DataLoader


class Trainer:
    """Enhanced trainer with early stopping and hyperparameter tuning"""

    def __init__(self, model, optimizer, early_stopping=None):
        self.model = model
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0

    def compute_accuracy(self, y_pred, y_true):
        """Compute classification accuracy"""
        y_pred_labels = np.argmax(y_pred.T, axis=1)
        y_true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
        return np.mean(y_pred_labels == y_true_labels)

    def train_epoch(self, X_train, y_train, batch_size, augmenter=None):
        """Train for one epoch with optional data augmentation"""
        epoch_losses = []
        epoch_accs = []

        for X_batch, y_batch in DataLoader.create_batches(X_train, y_train, batch_size):
            # Apply data augmentation if provided
            if augmenter:
                X_batch, y_batch = augmenter.augment_batch(X_batch, y_batch)

            # Forward propagation
            y_pred = self.model.forward_propagation(X_batch)

            # Compute loss and accuracy
            loss = self.model.compute_loss(y_pred, y_batch)
            acc = self.compute_accuracy(y_pred, y_batch)

            epoch_losses.append(loss)
            epoch_accs.append(acc)

            # Backward propagation and update
            gradients = self.model.backward_propagation(X_batch, y_batch)
            self.optimizer.update(self.model.parameters, gradients)

        return np.mean(epoch_losses), np.mean(epoch_accs)

    def train(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None,
              epochs=50, batch_size=32, verbose=True, augmenter=None):
        """Main training loop with early stopping"""

        self.model.training = True
        self.optimizer.initialize_parameters(self.model.parameters)

        for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
            # Training
            train_loss, train_acc = self.train_epoch(X_train, y_train, batch_size, augmenter)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # Validation
            self.model.training = False  # Turn off dropout for validation
            val_pred = self.model.forward_propagation(X_val)
            val_loss = self.model.compute_loss(val_pred, y_val)
            val_acc = self.compute_accuracy(val_pred, y_val)
            self.model.training = True  # Turn back on for training

            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Track best validation accuracy
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc

            # Test evaluation if provided
            if X_test is not None and y_test is not None and (epoch + 1) % 10 == 0:
                self.model.training = False
                test_pred = self.model.forward_propagation(X_test)
                test_acc = self.compute_accuracy(test_pred, y_test)
                self.model.training = True
            else:
                test_acc = None

            # Verbose output
            if verbose and (epoch + 1) % 5 == 0:
                log_msg = (f"Epoch {epoch +1}/{epochs} - "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                if test_acc is not None:
                    log_msg += f", Test Acc: {test_acc:.4f}"
                print(log_msg)

            # Early stopping check
            if self.early_stopping is not None:
                if self.early_stopping(val_loss, self.model):
                    print(f"Early stopping at epoch {epoch +1}")
                    break

        self.model.training = False  # Set to evaluation mode
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies

