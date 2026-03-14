import numpy as np


class AdamOptimizer:
    """Adam Optimizer with Learning rate scheduling"""

    def __init__(self, learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                 schedule_type='exponential', decay_rate=0.96, decay_steps=100):

        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.schedule_type = schedule_type
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.v = {}
        self.s = {}
        self.t = 0
        self.step_count = 0

    def initialize_parameters(self, parameters):
        """Initialize momentum terms"""
        for key in parameters.keys():
            self.v[key] = np.zeros_like(parameters[key])
            self.s[key] = np.zeros_like(parameters[key])

    def get_learning_rate(self):
        """Compute current learning rate based on schedule"""
        if self.schedule_type == 'exponential':
            ## Exponential decay
            self.learning_rate = self.initial_learning_rate * (self.decay_rate ** (self.step_count / self.decay_steps))
        elif self.schedule_type == 'step':
            ## Step decay
            self.learning_rate = self.initial_learning_rate * (self.decay_rate ** (self.step_count // self.decay_steps))
        elif self.schedule_type == 'inverse':
            ## Inverse time decay
            self.learning_rate = self.initial_learning_rate / ( 1 + self.decay_rate * (self.step_count / self.decay_steps))
        elif self.schedule_type == 'cosine':
            ## Cosine annealing
            self.learning_rate = self.initial_learning_rate * (0.5 * (1 +
                                                        np.cos(np.pi * self.step_count / self.decay_steps)))

        return self.learning_rate

    def update(self, parameters, gradients):
        self.t += 1
        self.step_count += 1

        current_lr = self.get_learning_rate()

        for key in parameters.keys():
            self.v[key] = self.beta_1 * self.v[key] + (1 - self.beta_1) * gradients['d' + key]
            self.s[key] = self.beta_2 * self.s[key] + (1 - self.beta_2) * (gradients['d' + key] ** 2)

            v_hat = self.v[key] / (1 - self.beta_1 ** self.t)
            s_hat = self.s[key] / (1 - self.beta_2 ** self.t)

            parameters[key] -= current_lr * v_hat / (np.sqrt(s_hat) + self.epsilon)

        return parameters

    def reset(self):
        """Reset optimizer state"""
        self.t = 0
        self.step_count = 0
        self.learning_rate = self.initial_learning_rate
        self.v = {}
        self.s = {}



