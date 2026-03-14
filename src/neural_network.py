import numpy as np


np.random.seed(42)

class NeuralNetwork:
    def __init__(self, layer_dims, activations, dropout_keep_prob=0.8, l2_lambda=0.01):
        self.layer_dims = layer_dims
        self.activations = activations
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_lambda = l2_lambda
        self.parameters = self.initialize_parameters()
        self.cache = {}
        self.training = True

    def initialize_parameters(self):
        parameters = {}

        for l in range(1, len(self.layer_dims)):
            if self.activations[l-1] == 'relu':
                scale = np.sqrt(2. / self.layer_dims[l-1])
            else:
                scale = np.sqrt(1. / self.layer_dims[l-1])

            parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * scale
            parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))

        return parameters

    def activation_function(self, Z, activation_type, derivative=False):
        if activation_type == 'relu':
            if derivative:
                return (Z > 0).astype(float)
            return np.maximum(0, Z)
        elif activation_type == 'sigmoid':
            sig = 1 / (1 + np.exp(-np.clip(Z, -250, 250)))
            return sig * (1 - sig) if derivative else sig
        elif activation_type == 'tanh':
            return 1 - np.tanh(Z)**2 if derivative else np.tanh(Z)
        elif activation_type == 'softmax':
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def forward_propagation(self, X):
        self.cache = {}
        A = X.T
        self.cache['A0'] = A

        for l in range(1, len(self.layer_dims)):
            Z = np.dot(self.parameters[f'W{l}'], A) + self.parameters[f'b{l}']

            if l == len(self.layer_dims) - 1:
                A = self.activation_function(Z, 'softmax')
            else:
                A = self.activation_function(Z, self.activations[l-1])

                if self.training and self.dropout_keep_prob < 1:
                    dropout_mask = np.random.rand(*A.shape) < self.dropout_keep_prob
                    A *= dropout_mask / self.dropout_keep_prob
                    self.cache[f'D{l}'] = dropout_mask

            self.cache[f'Z{l}'] = Z
            self.cache[f'A{l}'] = A

        return A


    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        y_true = y_true.T
        y_pred_clipped = np.clip(y_pred, 1e-8, 1 - 1e-8)
        cross_entropy = -np.sum(y_true * np.log(y_pred_clipped)) / m

        l2_loss = 0
        if self.l2_lambda > 0:
            for l in range(1, len(self.layer_dims)):
                l2_loss += np.sum(np.square(self.parameters[f'W{l}']))
            l2_loss *= (self.l2_lambda / (2 * m))

        return cross_entropy + l2_loss


    def backward_propagation(self, X, y_true):
        gradients = {}
        m = X.shape[0]
        y_true = y_true.T

        dZ = self.cache[f'A{len(self.layer_dims) - 1}'] - y_true

        for l in range(len(self.layer_dims) - 1, 0, -1):
            A_prev = self.cache[f'A{l - 1}']

            if l > 1 and l < len(self.layer_dims) - 1 and self.training and self.dropout_keep_prob < 1:
                A_prev *= self.cache[f'D{l-1}'] / self.dropout_keep_prob

            gradients[f'dW{l}'] = np.dot(dZ, A_prev.T) / m
            gradients[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / m

            if self.l2_lambda > 0:
                gradients[f'dW{l}'] += (self.l2_lambda / m) * self.parameters[f'W{l}']

            if l > 1:
                dA_prev = np.dot(self.parameters[f'W{l}'].T, dZ)
                dZ = dA_prev * self.activation_function(
                    self.cache[f'Z{l - 1}'],
                    self.activations[l - 2],
                    derivative=True
                )

        return gradients

