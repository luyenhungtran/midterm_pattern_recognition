
import numpy as np


class GaussianNB:

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None    #  (k,)
        self.priors_ = None    # (k,)
        self.means_ = None     #(k, d)
        self.vars_ = None      #(k, d)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n, d = X.shape

        self.classes_ = np.unique(y)
        k = len(self.classes_)

        self.priors_ = np.empty(k)
        self.means_ = np.empty((k, d))
        self.vars_ = np.empty((k, d))

        for i, c in enumerate(self.classes_):
            Xc = X[y == c]                      # các mẫu lớp c
            self.priors_[i] = Xc.shape[0] / n
            self.means_[i] = Xc.mean(axis=0)
            self.vars_[i] = Xc.var(axis=0) + self.var_smoothing
        return self

    def _log_joint(self, X):
        """Trả về ma trận shape (n, k): log P(y=c) + Σ_j log N(x_j | μ_c, σ²_c)."""
        # (n, 1, d) - (1, k, d) -> (n, k, d)
        diff = X[:, np.newaxis, :] - self.means_[np.newaxis, :, :]
        log_pdf = -0.5 * (
            np.log(2.0 * np.pi * self.vars_)[np.newaxis, :, :]
            + diff ** 2 / self.vars_[np.newaxis, :, :]
        )
        return log_pdf.sum(axis=2) + np.log(self.priors_)[np.newaxis, :]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        idx = self._log_joint(X).argmax(axis=1)
        return self.classes_[idx]
