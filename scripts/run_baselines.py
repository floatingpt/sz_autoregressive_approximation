#!/usr/bin/env python3
"""
Run MULTIVARIATE baseline methods and compare against MVAR classifier.

Strictly multivariate methods that use the full channel dimension (18 EEG channels).

Methods:
  - MultiRocketMultivariate: Multivariate rocket kernel transformer + RidgeClassifierCV
  - Arsenal: Multivariate kernel ensemble
  - STSForest: Supervised Time Series Forest (multivariate-capable)
  - CIF: Canonical Interval Forest (multivariate)
  - DrCIF: Diverse Representation CIF (multivariate)
  - Shapelet: Shapelet transform classifier (multivariate)
  - TimeCNN: 1D CNN (multivariate by design)
  - TimeRNN: LSTM (multivariate by design)

Usage:
  # Run all multivariate methods
  python scripts/run_baselines.py
  
  # Specific subset
  python scripts/run_baselines.py --methods MultiRocketMultivariate TimeCNN TimeRNN --epochs 5
  
  # With subsampling for speed
  python scripts/run_baselines.py --subsample 200 --epochs 3
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.data.paths import PATHS

# Optional deps
HAS_SKTIME = True
try:
    import pandas as pd
except Exception:
    HAS_SKTIME = False

HAS_TORCH = True
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except Exception:
    HAS_TORCH = False


def load_manifest(output_dir: Path) -> Dict:
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "r") as f:
        return json.load(f)


def load_data(output_dir: Path, subsample: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    manifest = load_manifest(output_dir)
    X_train = np.memmap(output_dir / "X_train.npy", dtype=np.float32, mode='r', shape=tuple(manifest['X_train_shape']))
    y_train = np.load(output_dir / "y_train.npy")
    X_test = np.memmap(output_dir / "X_test.npy", dtype=np.float32, mode='r', shape=tuple(manifest['X_test_shape']))
    y_test = np.load(output_dir / "y_test.npy")
    if subsample:
        # Downsample both splits independently to speed up baselines
        def _sub(x, y, n):
            if len(y) <= 2*n:
                return np.array(x), y
            idx0 = np.where(y == 0)[0]
            idx1 = np.where(y == 1)[0]
            n0 = min(n, len(idx0))
            n1 = min(n, len(idx1))
            rng = np.random.default_rng(42)
            keep = np.concatenate([rng.choice(idx0, size=n0, replace=False), rng.choice(idx1, size=n1, replace=False)])
            rng.shuffle(keep)
            return np.array(x[keep]), y[keep]
        X_train, y_train = _sub(X_train, y_train, subsample)
        X_test, y_test = _sub(X_test, y_test, subsample)
    return np.array(X_train), y_train, np.array(X_test), y_test, manifest


def dwt_features(X: np.ndarray, wavelet: str = "db4", level: Optional[int] = None) -> np.ndarray:
    raise NotImplementedError("Univariate methods removed. Use --methods with multivariate classifiers only.")


def lsw_features(X: np.ndarray, wavelet: str = "db4", level: Optional[int] = None, window: int = 64) -> np.ndarray:
    raise NotImplementedError("Univariate methods removed. Use --methods with multivariate classifiers only.")


def flogistic_features(X: np.ndarray, sr: int) -> np.ndarray:
    raise NotImplementedError("Univariate methods removed. Use --methods with multivariate classifiers only.")


def run_logistic_on_features(Xtr, Xte, ytr, yte) -> Dict[str, float]:
    raise NotImplementedError("Univariate feature-based methods removed. Use multivariate classifiers only.")


def to_sktime_nested(X: np.ndarray) -> "pd.DataFrame":
    if not HAS_SKTIME:
        raise RuntimeError("sktime not installed. Install with: pip install sktime")
    import pandas as pd
    # Each row is a sample; columns per channel; cells are pd.Series of length T
    n, c, t = X.shape
    data = {}
    for ch in range(c):
        data[f"ch_{ch}"] = [pd.Series(X[i, ch]) for i in range(n)]
    return pd.DataFrame(data)


def run_rocket_like(method: str, Xtr, Xte, ytr, yte) -> Dict[str, float]:
    if not HAS_SKTIME:
        raise RuntimeError("sktime not installed. Install with: pip install sktime")
    from sklearn.linear_model import RidgeClassifierCV
    n, c, t = Xtr.shape
    # Prefer multivariate-capable variants when channels > 1
    try:
        if c > 1:
            try:
                # Newer sktime
                from sktime.transformations.panel.rocket import MiniRocketMultivariate
                tr = MiniRocketMultivariate(random_state=42)
            except Exception:
                # Fallback older multivariate variant
                from sktime.transformations.panel.rocket import MultiRocketMultivariate
                tr = MultiRocketMultivariate(random_state=42)
        else:
            if method.lower() == "rocket":
                from sktime.transformations.panel.rocket import Rocket
                tr = Rocket(random_state=42)
            else:
                from sktime.transformations.panel.rocket import MultiRocket
                tr = MultiRocket(random_state=42)
    except Exception as e:
        raise RuntimeError(f"Could not instantiate a suitable Rocket transformer: {e}")

    Xtr_n = to_sktime_nested(Xtr)
    Xte_n = to_sktime_nested(Xte)
    tr.fit(Xtr_n)
    Xtr_f = tr.transform(Xtr_n)
    Xte_f = tr.transform(Xte_n)
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    clf.fit(Xtr_f, ytr)
    yhat = clf.predict(Xte_f)
    metrics = {"accuracy": float(accuracy_score(yte, yhat))}
    return metrics


def run_sktime_classifier(name: str, Xtr, Xte, ytr, yte) -> Dict[str, float]:
    if not HAS_SKTIME:
        raise RuntimeError("sktime not installed. Install with: pip install sktime")
    Xtr_n = to_sktime_nested(Xtr)
    Xte_n = to_sktime_nested(Xte)

    name_l = name.lower()
    clf = None
    try:
        if name_l == "arsenal":
            from sktime.classification.kernel_based import Arsenal
            clf = Arsenal(random_state=42)
        elif name_l in ("stsforest", "stsf"):
            # Multivariate-capable supervised time series forest
            from sktime.classification.interval_based import SupervisedTimeSeriesForest
            clf = SupervisedTimeSeriesForest(random_state=42)
        elif name_l in ("tsf", "timeseriesforest"):
            # Univariate-only TSF: collapse channels by averaging
            import pandas as pd
            from sktime.classification.interval_based import TimeSeriesForestClassifier

            def collapse_to_univariate(Xn):
                return pd.DataFrame({
                    "signal": [pd.concat([Xn.iloc[i, j] for j in range(Xn.shape[1])], axis=1).mean(axis=1)
                                for i in range(len(Xn))]
                })

            Xtr_u = collapse_to_univariate(Xtr_n)
            Xte_u = collapse_to_univariate(Xte_n)
            clf = TimeSeriesForestClassifier(random_state=42)
            clf.fit(Xtr_u, ytr)
            yhat = clf.predict(Xte_u)
            return {"accuracy": float(accuracy_score(yte, yhat))}
        elif name_l in ("shapelet", "shapelettransformclassifier"):
            from sktime.classification.shapelet_based import ShapeletTransformClassifier
            clf = ShapeletTransformClassifier(time_limit_in_minutes=3, random_state=42)
        elif name_l in ("cif", "canonicalintervalforest"):
            from sktime.classification.interval_based import CanonicalIntervalForest
            clf = CanonicalIntervalForest(random_state=42)
        elif name_l in ("drcif", "drcanonicalintervalforest"):
            from sktime.classification.interval_based import DrCIF
            clf = DrCIF(random_state=42)
        else:
            raise ValueError(f"Unknown sktime classifier: {name}")
    except Exception as e:
        raise RuntimeError(f"Could not instantiate {name}: {e}")

    clf.fit(Xtr_n, ytr)
    yhat = clf.predict(Xte_n)
    metrics = {"accuracy": float(accuracy_score(yte, yhat))}
    return metrics


# Torch models (only define if torch is available)
if HAS_TORCH:
    class TimeCNN(nn.Module):
        def __init__(self, n_channels: int, n_time: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.fc = nn.Linear(64, 1)

        def forward(self, x):
            h = self.net(x).squeeze(-1)
            return self.fc(h).squeeze(-1)


    class TimeRNN(nn.Module):
        def __init__(self, n_channels: int, hidden: int = 64, layers: int = 1):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_channels, hidden_size=hidden, num_layers=layers, batch_first=True)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            # x: [B, C, T] → [B, T, C]
            x = x.transpose(1, 2)
            out, _ = self.lstm(x)
            h = out[:, -1, :]
            return self.fc(h).squeeze(-1)
else:
    # Placeholder for when torch isn't available
    class TimeCNN:
        pass
    class TimeRNN:
        pass


def run_torch_model(kind: str, Xtr, Xte, ytr, yte, epochs: int = 5, batch_size: int = 64, lr: float = 1e-3) -> Dict[str, float]:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not installed. Install with: conda install pytorch torchvision torchaudio -c pytorch or pip install torch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, c, t = Xtr.shape
    if kind == "TimeCNN":
        model = TimeCNN(c, t)
    else:
        model = TimeRNN(c)
    model.to(device)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32).to(device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).to(device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32).to(device)
    yte_t = torch.tensor(yte, dtype=torch.float32).to(device)
    ds = TensorDataset(Xtr_t, ytr_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(Xte_t)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(yte, preds)),
        "roc_auc": float(roc_auc_score(yte, probs)),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run multivariate baseline classifiers to compare with MVAR classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all multivariate methods
  python scripts/run_baselines.py

  # Specific subset
  python scripts/run_baselines.py --methods MultiRocket Arsenal STSForest CIF DrCIF Shapelet TimeCNN TimeRNN

  # Fast run with subsampling
  python scripts/run_baselines.py --subsample 100 --epochs 3

Methods (all multivariate):
  - Rocket/MultiRocket: Kernel-based transformers
  - Arsenal: Kernel ensemble
  - STSForest: Supervised time series forest
  - CIF: Canonical Interval Forest
  - DrCIF: Diverse Representation CIF
  - Shapelet: Shapelet-based classifier
  - TimeCNN: 1D convolutional neural network
  - TimeRNN: LSTM recurrent neural network
        """
    )
    parser.add_argument("--output-dir", type=Path, default=PATHS["features"], help="Directory containing manifest and saved arrays")
    parser.add_argument("--methods", nargs="+", default=["MultiRocket", "Arsenal", "STSForest", "CIF", "DrCIF", "Shapelet", "TimeCNN", "TimeRNN"], 
                        help="List of multivariate methods to run")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for TimeCNN/TimeRNN")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for TimeCNN/TimeRNN")
    parser.add_argument("--subsample", type=int, default=None, help="Per-class subsample size to speed up baselines")
    args = parser.parse_args()

    # Optional reproducibility
    try:
        np.random.seed(42)
    except Exception:
        pass

    X_train, y_train, X_test, y_test, manifest = load_data(args.output_dir, subsample=args.subsample)
    sr = int(manifest.get("sample_rate", 256))

    # Diagnostics
    try:
        uniq_tr, cnt_tr = np.unique(y_train, return_counts=True)
        uniq_te, cnt_te = np.unique(y_test, return_counts=True)
        print("Train distribution:", dict(zip(uniq_tr.tolist(), cnt_tr.tolist())))
        print("Test distribution:", dict(zip(uniq_te.tolist(), cnt_te.tolist())))
        print(f"Data shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    except Exception:
        pass

    results = {}
    for m in args.methods:
        try:
            m_lower = m.lower()
            if m_lower in ("rocket", "multirocket", "minirocketmultivariate", "multirocketmultivariate"):
                results[m] = run_rocket_like(m, X_train, X_test, y_train, y_test)
            elif m_lower in ("arsenal", "stsforest", "stsf", "tsf", "shapelet", "cif", "canonicalintervalforest", "drcif", "drcanonicalintervalforest"):
                # Map alias to canonical name
                canonical_map = {
                    "arsenal": "Arsenal",
                    "stsforest": "STSForest",
                    "stsf": "STSForest",
                    "tsf": "TSF",
                    "shapelet": "Shapelet",
                    "cif": "CIF",
                    "canonicalintervalforest": "CIF",
                    "drcif": "DrCIF",
                    "drcanonicalintervalforest": "DrCIF",
                }
                name = canonical_map.get(m_lower, m)
                results[m] = run_sktime_classifier(name, X_train, X_test, y_train, y_test)
            elif m_lower in ("timecnn", "timernn"):
                results[m] = run_torch_model("TimeCNN" if m_lower == "timecnn" else "TimeRNN",
                                             X_train, X_test, y_train, y_test,
                                             epochs=args.epochs, batch_size=args.batch_size)
            else:
                results[m] = {"error": f"Unknown multivariate method: {m}. Use --methods with: Rocket, MultiRocket, Arsenal, STSForest, CIF, DrCIF, Shapelet, TimeCNN, TimeRNN"}
        except Exception as e:
            results[m] = {"error": str(e)}

    out_dir = PATHS["results"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ Multivariate Baseline Results")
    print(f"{'='*80}")
    print(f"Saved to: {out_path}\n")
    
    # Print summary
    for k, v in results.items():
        if isinstance(v, dict) and "error" in v:
            print(f"  {k:20s} ERROR: {v['error']}")
        elif isinstance(v, dict):
            acc = v.get("accuracy", "N/A")
            auc = v.get("roc_auc", "N/A")
            print(f"  {k:20s} Accuracy: {acc:.4f}  |  AUC: {auc}")
        else:
            print(f"  {k:20s} {v}")
    
    print(f"\nCompare these results with your MVAR classifier performance.")
    print(f"All methods are strictly multivariate (use all {X_train.shape[1]} channels).\n")

if __name__ == "__main__":
    main()