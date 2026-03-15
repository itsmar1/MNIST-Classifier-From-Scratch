# Model configuration
MODEL_CONFIG = {
    'input_size': 784,
    'hidden_sizes': [128, 64],
    'output_size': 10,
    'layer_dims': [784, 128, 64, 10],
    'activations': ['relu', 'relu', 'softmax'],
    'dropout_keep_prob': 0.8,
    'l2_lambda': 0.001
}

# Training configuration
TRAINING_CONFIG = {
    'epochs': 50,
    'batch_size': 64,
    'learning_rate': 0.001,
    'validation_split': 0.15,
    'test_split': 0.1,
    'use_augmentation': True
}

# Optimizer configuration
OPTIMIZER_CONFIG = {
    'learning_rate': 0.001,
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
    'schedule_type': 'exponential',
    'decay_rate': 0.96,
    'decay_steps': 100
}

EARLY_STOPPING_CONFIG = {
    'patience': 10,
    'min_delta': 0.001,
    'restore_best_weights': True
}

# Paths
MODEL_PATH = 'models/'