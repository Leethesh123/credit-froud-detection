import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim=16, lr=0.01, pos_weight=1.0):
        self.lr = lr
        self.pos_weight = pos_weight

        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def relu(self, z):
        return np.maximum(0, z)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.y_hat = self.sigmoid(self.z2)
        return self.y_hat

    def compute_loss(self, y, y_hat):
        y = y.reshape(-1, 1)
        eps = 1e-8

        loss = -(
            self.pos_weight * y * np.log(y_hat + eps)
            + (1 - y) * np.log(1 - y_hat + eps)
        )
        return np.mean(loss)

    def backward(self, X, y):
        m = X.shape[0]
        y = y.reshape(-1, 1)

        dz2 = (self.y_hat - y)
        dz2[y == 1] *= self.pos_weight

        dW2 = self.a1.T @ dz2 / m
        db2 = np.mean(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)

        dW1 = X.T @ dz1 / m
        db1 = np.mean(dz1, axis=0, keepdims=True)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
