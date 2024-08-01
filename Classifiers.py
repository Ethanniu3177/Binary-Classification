import numpy as np


class WineModel:
    def __init__(self, in_features):
        # We set initial weights to small normally distributed values.
        self.w = 0.01 * np.random.randn(in_features)
        self.w_0 = 0.01 * np.random.randn()
        self.non_zero_tolerance = (
            1e-8  # add this to divisions to ensure we don't divide by 0
        )

    def forward(self, x):
        # Calculate and save the pre-activation z
        self.z = x @ self.w.T + self.w_0

        # Apply the activation function, and return
        self.a = self.activation(self.z)
        return self.a

    # update weights based on gradients and learning rate
    def update(self, grad_loss, learning_rate):
        self.w -= grad_loss * self.grad_w * learning_rate
        self.w_0 -= grad_loss * self.grad_w_0 * learning_rate


# New implementation! Single neuron classification model
class WineClassificationModel(WineModel):
    # Sigmoid activation function for classification
    def activation(self, z):
        return 1 / (1 + np.exp(-z) + self.non_zero_tolerance)

    # Gradient of output w.r.t. weights, for sigmoid activation
    def gradient(self, x):
        self.grad_w = self.a * (1 - self.a) * x
        self.grad_w_0 = self.a * (1 - self.a)
