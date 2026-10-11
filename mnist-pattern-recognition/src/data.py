import gzip, hashlib, json, urllib.request
import numpy as np
from .config import DATA_DIR, LOG_DIR
"""
gzip dùng để giải nén file
hashlib để tính mã băm SHA256
json dùng để lưu file thống kê
urllib.request để tải file theo link URL
"""

BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}
RAW_DIR = DATA_DIR / "raw"

def _download(name):
    path = RAW_DIR/ name
    if not path.exists():
        RAW_DIR.mkdir(parents=True, exist_ok = True)
        urllib.request.urlretrieve(BASE_URL + name, path)
    return path

def _read_images(path):
    data = gzip.open(path, "rb").read()
    magic, n, rows, cols = [int(v) for v in np.frombuffer(data[:16], dtype = ">i4")]
    assert magic == 2051, "File ảnh không đúng định dạng IDX"
    return np.frombuffer(data, dtype = np.uint8, offset = 16).reshape(n, rows, cols).copy()

def _read_labels(path):
    data = gzip.open(path, "rb").read()
    magic, n = [int(v) for v in np.frombuffer(data[:8], dtype = ">i4")]
    assert magic == 2049, "File nhãn không đúng định dạng IDX"
    return np.frombuffer(data, dtype = np.uint8, offset = 8).copy()

def load_mnist():
    #Trả về X_train, y_train, X_test, y_test
    p = {k: _download(v) for k, v in FILES.items()}
    return (
        _read_images(p["train_images"]),
        _read_labels(p["train_labels"]),
        _read_images(p["test_images"]),
        _read_labels(p["test_labels"])
    )
