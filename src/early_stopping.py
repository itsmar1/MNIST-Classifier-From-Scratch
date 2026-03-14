

class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        """
        Initialize early stopping

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Restore best model weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        """Check if training should stop"""
        if self.best_score is None:
            self.best_score = val_loss
            if self.restore_best_weights:
                self.best_weights = {k: v.copy() for k, v in model.parameters.items()}

        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    model.parameters = self.best_weights
                print(f"\nEarly stopping triggered! Restored best model from epoch {self.counter}")
        else:
            self.best_score = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.copy() for k, v in model.parameters.items()}

        return self.early_stop