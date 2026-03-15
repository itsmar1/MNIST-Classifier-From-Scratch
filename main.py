#!/usr/bin/env python3
"""
Main entry point for training the handwritten digit classifier
"""
import argparse
import os
import pickle
import numpy as np
import config
from src.data_loader import DataLoader
from src.data_augmenter import DataAugmenter
from src.neural_network import NeuralNetwork
from src.adam_optimizer import AdamOptimizer
from src.early_stopping import EarlyStopping
from src.trainer import Trainer
from src.visualizer import Visualizer


def main():
    # Load data
    print("Loading data...")
    X, y = DataLoader.load_mnist()

    # Preprocess
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     y_train_orig, y_val_orig, y_test_orig) = DataLoader.preprocess_data(
        X, y, validation_split=config.TRAINING_CONFIG['validation_split'], test_split=config.TRAINING_CONFIG['test_split']
    )

    # Create model
    model = NeuralNetwork(
        layer_dims=config.MODEL_CONFIG['layer_dims'],
        activations=config.MODEL_CONFIG['activations'],
        dropout_keep_prob=config.MODEL_CONFIG['dropout_keep_prob'],
        l2_lambda=config.MODEL_CONFIG['l2_lambda']
    )

    # Create optimizer
    optimizer = AdamOptimizer(
        learning_rate=config.OPTIMIZER_CONFIG['learning_rate'],
        beta_1=config.OPTIMIZER_CONFIG['beta1'],
        beta_2=config.OPTIMIZER_CONFIG['beta2'],
        epsilon=config.OPTIMIZER_CONFIG['epsilon'],
        schedule_type=config.OPTIMIZER_CONFIG['schedule_type'],
        decay_rate=config.OPTIMIZER_CONFIG['decay_rate'],
        decay_steps=config.OPTIMIZER_CONFIG['decay_steps']
    )

    # Create early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_CONFIG['patience'],
        min_delta=config.EARLY_STOPPING_CONFIG['min_delta'],
        restore_best_weights=config.EARLY_STOPPING_CONFIG['restore_best_weights']
    )

    # Create augmenter if needed
    augmenter = DataAugmenter() if config.TRAINING_CONFIG['use_augmentation'] else None

    # Train
    trainer = Trainer(model, optimizer, early_stopping)
    trainer.train(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test_orig,
        epochs=config.TRAINING_CONFIG['epochs'],
        batch_size=config.TRAINING_CONFIG['batch_size'],
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




if __name__ == "__main__":
    main()