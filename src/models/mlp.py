import math

import numpy as np

from ..metrics import one_hot, softmax


class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, activation="relu", weight_scale=None, seed=42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        rng = np.random.default_rng(seed)

        if weight_scale is None:
            if activation == "relu":
                w1_scale = math.sqrt(2.0 / input_dim)
            else:
                w1_scale = math.sqrt(1.0 / input_dim)
            w2_scale = math.sqrt(1.0 / hidden_dim)
        else:
            w1_scale = weight_scale
            w2_scale = weight_scale

        self.params = {
            "W1": rng.standard_normal((input_dim, hidden_dim), dtype=np.float32) * w1_scale,
            "b1": np.zeros((1, hidden_dim), dtype=np.float32),
            "W2": rng.standard_normal((hidden_dim, output_dim), dtype=np.float32) * w2_scale,
            "b2": np.zeros((1, output_dim), dtype=np.float32),
        }

    def activate(self, x):
        if self.activation == "relu":
            return np.maximum(0, x)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        if self.activation == "tanh":
            return np.tanh(x)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def activation_grad(self, x):
        if self.activation == "relu":
            return (x > 0).astype(np.float32)
        if self.activation == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-x))
            return s * (1.0 - s)
        if self.activation == "tanh":
            t = np.tanh(x)
            return 1.0 - t * t
        raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(self, X):
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        z1 = X @ W1 + b1
        a1 = self.activate(z1)
        logits = a1 @ W2 + b2
        cache = {"X": X, "z1": z1, "a1": a1, "logits": logits}
        return logits, cache

    def loss_and_grads(self, X, y, weight_decay=0.0):
        logits, cache = self.forward(X)
        probs = softmax(logits)
        y_oh = one_hot(y, self.output_dim)
        batch_size = X.shape[0]

        ce_loss = -np.sum(y_oh * np.log(probs + 1e-12)) / batch_size
        l2_loss = 0.5 * weight_decay * (
            np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2)
        )
        loss = ce_loss + l2_loss

        dlogits = (probs - y_oh) / batch_size
        dW2 = cache["a1"].T @ dlogits + weight_decay * self.params["W2"]
        db2 = np.sum(dlogits, axis=0, keepdims=True)
        da1 = dlogits @ self.params["W2"].T
        dz1 = da1 * self.activation_grad(cache["z1"])
        dW1 = cache["X"].T @ dz1 + weight_decay * self.params["W1"]
        db1 = np.sum(dz1, axis=0, keepdims=True)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return loss, grads, logits

    def predict(self, X):
        logits, _ = self.forward(X)
        return np.argmax(logits, axis=1)

    def update(self, grads, lr):
        for key in self.params:
            self.params[key] -= lr * grads[key]
