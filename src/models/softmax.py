import numpy as np

class SoftmaxRegression:
    def __init__(self, learning_rate=0.1, epochs=100, batch_size=256,
                 l2=0.0, random_state=42, patience=15, verbose=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.random_state = random_state
        self.patience = patience
        self.verbose = verbose

    @staticmethod
    def softmax(logits):
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def _loss(self, X, y):
        z = X @ self.W_ + self.b_
        z = z - z.max(axis=1, keepdims=True)
        ce = np.mean(np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(y)), y])
        return float(ce)

    def _grad(self, X, y):
        error = self.softmax(X @ self.W_ + self.b_)
        error[np.arange(len(y)), y] -= 1
        dW = X.T @ error / len(y) + self.l2 * self.W_
        db = error.mean(axis=0)
        return dW, db

    def fit(self, X, y, X_val=None, y_val=None):
        if self.learning_rate <= 0 or self.epochs < 1 or self.batch_size < 1 or self.l2 < 0:
            raise ValueError('learning_rate, epochs, batch_size phải dương; l2 >= 0.')
        if (X_val is None) != (y_val is None):
            raise ValueError('X_val và y_val phải đồng thời tồn tại.')
        X = np.asarray(X)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.arange(10)

        self.W_ = np.zeros((X.shape[1], len(self.classes_)), dtype=np.float32)
        self.b_ = np.zeros(len(self.classes_), dtype=np.float32)
        if X_val is not None:
            y_val = np.asarray(y_val, dtype=int)
        rng = np.random.default_rng(self.random_state)
        self.history_ = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
        best_loss, stale, best = np.inf, 0, None
        self.best_epoch_ = None
        for epoch in range(self.epochs):
            indices = rng.permutation(len(X))
            for start in range(0, len(X), self.batch_size):
                idx = indices[start:start + self.batch_size]
                dW, db = self._grad(X[idx], y[idx])
                self.W_ -= self.learning_rate * dW
                self.b_ -= self.learning_rate * db
            train_loss = self._loss(X, y)
            train_acc = self.score(X, y)
            self.history_['train_loss'].append(train_loss)
            self.history_['train_accuracy'].append(train_acc)
            if X_val is not None:
                val_loss = self._loss(X_val, y_val)
                self.history_['val_loss'].append(val_loss)
                self.history_['val_accuracy'].append(self.score(X_val, y_val))
                if val_loss < best_loss - 1e-6:
                    best_loss, stale = val_loss, 0
                    best = (self.W_.copy(), self.b_.copy())
                    self.best_epoch_ = epoch + 1
                else:
                    stale += 1
            self.n_iter_ = epoch + 1
            if self.verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
                msg = f'Epoch {epoch + 1:3d} | train CE={train_loss:.4f} | acc={train_acc:.4f}'
                if X_val is not None:
                    msg += f' | val CE={val_loss:.4f}'
                print(msg)
            if X_val is not None and self.patience is not None and stale >= self.patience:
                break
        if best is not None:
            self.W_, self.b_ = best
        return self

    def predict_proba(self, X):
        return self.softmax(np.asarray(X) @ self.W_ + self.b_)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))