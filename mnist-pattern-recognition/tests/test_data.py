import numpy as np
from src.data import load_mnist

def test_mnist_contract():
    X_tr, y_tr, X_te, y_te = load_mnist()
    assert X_tr.shape == (60000, 28, 28)
    assert X_te.shape == (10000, 28, 28)
    assert y_tr.shape == (60000,) and y_te.shape == (10000,)
    assert X_tr.dtype == np.uint8
    assert X_tr.min() >= 0 and X_tr.max() <=255
    assert set(np.unique(y_tr)) == set(range(10))
    assert set(np.unique(y_te)) == set(range(10))



    