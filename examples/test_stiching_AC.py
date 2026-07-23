import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import scarlet.reduction.stitching as st

def load_txt_curve(path: Path) -> st.SASCurve:
    """Load one ASCII I(Q) export with columns Q, I, dI, dQ."""
    data = np.loadtxt(path)
    data = data[data[:, 1] > 0]
    return st.SASCurve(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
        name=path.stem,
        config_id=path.stem,
    )

data_dir = Path(__file__).resolve().parent / "data_SANS_LLB"
curve_paths = (
    data_dir / "ludox_AM20_config_config_9_detector0.txt",
    data_dir / "ludox_AM20_config_config_10_detector0.txt",
)
curves = tuple(sorted((load_txt_curve(path) for path in curve_paths), key=lambda curve: curve.q.min()))



