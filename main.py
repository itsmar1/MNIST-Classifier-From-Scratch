#!/usr/bin/env python3
"""
Main entry point for training the handwritten digit classifier
"""
import argparse
import os
import pickle
import numpy as np
from src.data_loader import DataLoader
from src.data_augmenter import DataAugmenter
from src.neural_network import NeuralNetwork
from src.adam_optimizer import AdamOptimizer
from src.early_stopping import EarlyStopping
from src.trainer import Trainer
from src.visualizer import Visualizer


def main(args):
    # Load data
    print("Loading data...")
    X, y = DataLoader.load_mnist()

    # Preprocess
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     y_train_orig, y_val_orig, y_test_orig) = DataLoader.preprocess_data(
        X, y, validation_split=0.15, test_split=0.1
    )

    # Create model
    model = NeuralNetwork(
        layer_dims=[784, args.hidden_size, args.hidden_size // 2, 10],
        activations=[args.activation, args.activation, 'softmax'],
        dropout_keep_prob=args.dropout,
        l2_lambda=args.l2_lambda
    )

    # Create optimizer
    optimizer = AdamOptimizer(
        learning_rate=args.learning_rate,
        schedule_type=args.schedule,
        decay_rate=0.96
    )

    # Create early stopping
    early_stopping = EarlyStopping(patience=args.patience)

    # Create augmenter if needed
    augmenter = DataAugmenter() if args.augment else None

    # Train
    trainer = Trainer(model, optimizer, early_stopping)
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test_orig,
        epochs=args.epochs,
        batch_size=args.batch_size,
        augmenter=augmenter
    )

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Save model
    with open('models/trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Visualize
    Visualizer.plot_training_history(
        trainer.train_losses,
        trainer.val_losses,
        trainer.train_accuracies,
        trainer.val_accuracies
    )

    return model, trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train digit classifier')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.8)
    parser.add_argument('--l2_lambda', type=float, default=0.001)
    parser.add_argument('--schedule', type=str, default='exponential')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--augment', action='store_true')

    args = parser.parse_args()
    main(args)