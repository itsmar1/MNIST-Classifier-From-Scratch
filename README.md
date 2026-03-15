Handwritten Digit Classifier from Scratch with NumPy
====================================================

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A complete implementation of a neural network for handwritten digit recognition (MNIST) using only NumPy - no deep learning frameworks like PyTorch or TensorFlow. Features the Adam optimization algorithm, data augmentation, regularization techniques, and achieves **96.10% validation accuracy**.

Model Performance
-----------------

*   **Validation Accuracy**: 96.10%
    
*   **Architecture**: 784 → 128 → 64 → 10
    
*   **Optimizer**: Adam with learning rate scheduling
    
*   **Regularization**: Dropout (0.8 keep probability) + L2 regularization
    

Features
--------

### Core Implementation

*   **Pure NumPy**: No deep learning frameworks, everything built from scratch
    
*   **Modular Design**: Clean, reusable components with proper separation of concerns
    
*   **Configurable Architecture**: Easily change layer sizes, activations, etc.
    

### Advanced Optimizations

*   **Adam Optimizer**: Complete implementation with bias correction
    
*   **Learning Rate Scheduling**: Exponential, step, cosine, and inverse time decay
    
*   **Early Stopping**: Prevents overfitting with patience mechanism
    
*   **Gradient Clipping**: Prevents exploding gradients
    

### Regularization Techniques

*   **Dropout**: Randomly drops neurons during training (configurable keep probability)
    
*   **L2 Regularization**: Weight decay to prevent overfitting
    
*   **He/Xavier Initialization**: Proper weight initialization based on activation functions
    

### Data Processing

*   **Data Augmentation**: Shift, noise, and rotation augmentations
    
*   **Mini-batch Training**: Efficient batch processing
    
*   **Automatic Preprocessing**: Normalization and one-hot encoding
    

### Visualization & Monitoring

*   **Training Curves**: loss and accuracy plots
    

Project Structure
-----------------
```text
handwritten_digit_classifier/
│
├── src/
│   ├── adam_optimizer.py        # Adam optimizer with scheduling
│   ├── data_augmenter.py        # Data augmentation techniques
│   ├── data_loader.py           # MNIST loading and preprocessing
│   ├── early_stopping.py        # Early stopping implementation
│   ├── neural_network.py        # Core neural network architecture
│   ├── trainer.py               # Training loop and logic
│   └── visualizer.py            # Plotting and visualization
│
├── models/                      # Saved model checkpoints
├── config.py                    # Centralized configuration
├── main.py                      # Main training script
├── requirements.txt             # Dependencies
├── plot.PNG                     # Training results plot
└── README.md                    # This file
```

Getting Started
---------------

### Prerequisites

*   Python 3.8 or higher
    
*   pip package manager
    

### Installation

1. **Clone the repository**
    
```bash
git clone https://github.com/itsmar1/MNIST-Classifier-From-Scratch.git  
cd handwritten-digit-classifier
```
2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```
### Quick Start

Run the training with default settings:
```bash
python main.py
```

### Configuration Options

The project uses a centralized config.py file for all settings.

Results and Visualization
-------------------------

The plot shows excellent convergence with:
![Traning and Validation Loss and Accuracy](https://github.com/user-attachments/assets/6f82d63c-6f21-4070-b753-015fd40f69ea)
*   Final validation accuracy: **96.10%**
    
*   Smooth loss curves with minimal overfitting
    
*   Stable training progression

        

Contributing
------------

Contributions are welcome! Feel free to:

*   Open issues for bugs or suggestions
    
*   Submit pull requests with improvements
    

License
-------

This project is licensed under the MIT License - see the `LICENSE` file for details.