import os
from pathlib import Path
import numpy as np
from typing import Optional, Tuple
import mat73

try:
    DATA_ROOT = Path(__file__).parent
except NameError:
    DATA_ROOT = Path(os.getcwd())


def get_np_dir():
    candidate = DATA_ROOT / "Npy_files"
    if candidate.exists() and candidate.is_dir():
        return candidate

    # fallback: search for directory named "Npy_files" anywhere under DATA_ROOT
    for root, dirs, files in os.walk(DATA_ROOT):
        if "Npy_files" in dirs:
            return Path(root) / "Npy_files"

    print("Npy_files directory not found. Please ensure correct file structure.")
    return None


def find_train_files(np_dir: Path, kind: str = "train"):
    files = list(np_dir.glob("*.npy"))
    matched = [f for f in files if kind in f.name]
    if not matched:
        raise FileNotFoundError(f"No '{kind}' .npy files found in {np_dir}")

    x_file = next((f for f in matched if f.name.lower().startswith("x")), None)
    y_file = next((f for f in matched if f.name.lower().startswith("y")), None)

    if not x_file or not y_file:
        # try a looser heuristic: file name contains 'x' or 'y'
        for f in matched:
            name = f.name.lower()
            if "x" in name and x_file is None:
                x_file = f
            if "y" in name and y_file is None:
                y_file = f

    if not x_file or not y_file:
        raise FileNotFoundError(
            f"Could not determine x/y files among: {[f.name for f in matched]} in {np_dir}"
        )

    return x_file, y_file


def load_train(np_dir: Optional[Path]) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(np_dir, list):
        np_dir = np_dir[0] if np_dir else None
    if np_dir is None:
        raise FileNotFoundError("np_dir is None. Call get_np_dir() to locate the Npy_files directory.")
    np_dir = Path(np_dir)

    x_file, y_file = find_train_files(np_dir, "train")
    return np.load(x_file), np.load(y_file)


def load_test(np_dir: Optional[Path]) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(np_dir, list):
        np_dir = np_dir[0] if np_dir else None
    if np_dir is None:
        raise FileNotFoundError("np_dir is None. Call get_np_dir() to locate the Npy_files directory.")
    np_dir = Path(np_dir)

    x_file, y_file = find_train_files(np_dir, "test")
    return np.load(x_file), np.load(y_file)
