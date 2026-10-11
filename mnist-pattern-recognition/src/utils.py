import json, platform, random, sys
import numpy as np
from .config import SEED, LOG_DIR

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

def log_environment():
    import matplotlib
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "seed": SEED,
        "device": "cpu",   # đổi thành "cuda" nếu dùng GPU
    }
    (LOG_DIR / "environment.json").write_text(json.dumps(info, indent=2))
    return info