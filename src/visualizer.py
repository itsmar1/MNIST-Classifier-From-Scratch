from matplotlib import pyplot as plt


class Visualizer:
    """Enhanced visualization with more options"""

    @staticmethod
    def plot_training_history(train_losses, val_losses, train_accs, val_accs,
                              learning_rates=None):
        """Plot training metrics with optional learning rate"""
        n_plots = 3 if learning_rates is not None else 2
        fig, axes = plt.subplots(1, n_plots, figsize=( 5 *n_plots, 4))

        # Plot losses
        axes[0].plot(train_losses, label='Train Loss', alpha=0.8)
        axes[0].plot(val_losses, label='Validation Loss', alpha=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot accuracies
        axes[1].plot(train_accs, label='Train Accuracy', alpha=0.8)
        axes[1].plot(val_accs, label='Validation Accuracy', alpha=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot learning rate if provided
        if learning_rates is not None and n_plots > 2:
            axes[2].plot(learning_rates)
            axes[2].set_xlabel('Step')
            axes[2].set_ylabel('Learning Rate')
            axes[2].set_title('Learning Rate Schedule')
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

