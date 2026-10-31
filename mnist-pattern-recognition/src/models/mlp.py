import numpy as np

def _softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=-1, keepdims=True)

class MLPClassifier:
    #Dùng ReLU - Softmax, dùng cross-entropy
    def __init__(
            self,
            hidden_units: tuple[int, ...] = (256,128),
            learning_rate: float = 0.08,
            epochs: int = 20,
            batch_size: int = 256,
            l2: float = 1e-4,
            seed: int = 42
    ):
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.seed = seed
        self.history_: list[float] = []

    def _init_para(self, n_features, n_classes, generator):
        sizes = [n_features, *self.hidden_units, n_classes]
        self.weights_ = [generator.normal(loc = 0, scale = np.sqrt(2/sizes[i]), size =(sizes[i], sizes[i+1])) for i in range(len(sizes) -1)]
        self.biases_ = [np.zeros(sizes[i+1]) for i in range (len(sizes) - 1) ]

    def _forward(self, X):
        activations = [X]
        for W,b in zip(self.weights_[:-1], self.biases_[:-1]):
            activations.append(np.maximum((activations[-1] @ W + b),0.0))
        prob = _softmax(activations[-1] @ self.weights_[-1] + self.biases_[-1])
        return activations, prob

    def _backward(self, X_batch, y_batch):
        activations, prob = self._forward(X_batch)
        grad = prob.copy()

        # Trừ 1 tại vị trí lớp thật của mỗi mẫu, tương đương grad = prob - y_batch_OHE
        grad[np.arange(len(y_batch)), y_batch] -= 1.0
        grad /= len(y_batch)

        for i in reversed(range(len(self.weights_))):
            grad_W = activations[i].T @ grad + self.l2 * self.weights_[i]
            grad_b = grad.sum(axis = 0)
            if i > 0:
                grad = (grad @ self.weights_[i].T) * (activations[i] > 0)
            self.weights_[i] -= self.learning_rate * grad_W
            self.biases_[i] -= self.learning_rate * grad_b

    def _loss(self, X, y):
        _, prob = self._forward(X)
        ce = -np.mean(np.log(prob[np.arange(len(y)), y] + 1e-12))
        reg = 0.5 * self.l2 * sum(np.sum(W**2) for W in self.weights_)
        return ce + reg

    def fit(self, X, y):
        X = np.asarray(X, dtype = np.float64)
        self.classes_, y_idx = np.unique(y, return_inverse= True)
        rng = np.random.default_rng(self.seed)
        self._init_para(X.shape[1], len(self.classes_), rng )
        self.history_ = []

        n = len(X)
        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                self._backward(X[idx], y_idx[idx])
            loss = self._loss(X, y_idx)
            self.history_.append(loss)
        return self

    def predict_proba(self, X):
        return self._forward(np.asarray(X, dtype=np.float64))[1]

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y): # Số lần dự đoán đúng/ tổng số mẫu
        return float(np.mean(self.predict(X) == np.asarray(y)))