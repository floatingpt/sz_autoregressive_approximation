# src/train.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from joblib import dump
from var_model import var_features_from_A_Var
import scipy as sc
import pandas as pd

import os
from pathlib import Path
import sys

try:
    PROJECT_ROOT = Path(__file__).parent.parent
except NameError:
    PROJECT_ROOT = Path(os.getcwd())

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.load_npy_dat import get_np_dir, load_train, load_test


def vectorized_var_fit(X_all, p=5, ridge=1e-6):    # (N, channels, T) -> (N, p, channels, channels), Var_all (N, channels, channels)
    N, n_channels, T = X_all.shape
    if T <= p: 
        raise ValueError("Time dimension must be > order p")

    #  -> (N, T, n)
    data = np.transpose(X_all, (0, 2, 1))
    # lagged design matrix (N, T-p, n_channels * p)
    lagged = []
    for k in range(p):
        lagged.append(data[:, p - k - 1:T - k - 1, :])
    X_stack = np.concatenate(lagged, axis=2)  # (N, T-p, n*p)
    Y = data[:, p:, :]  # (N, T-p, n)

    #normeq
    # XtX: (N, n*p, n*p), XtY: (N, n*p, n)
    Xt = np.transpose(X_stack, (0, 2, 1))
    XtX = np.matmul(Xt, X_stack)
    XtY = np.matmul(Xt, Y)

    # regularize XtX
    M = XtX.shape[1]
    I = np.eye(M)[None, ...]
    XtX_reg = XtX + ridge * I

    # solve for B: (N, n*p, n)
    B = np.linalg.solve(XtX_reg, XtY)

    # reshape to A_all (N, p, n, n)
    n = n_channels
    B_resh = B.reshape(N, n, p, n).transpose(0, 2, 1, 3)
    A_all = B_resh.copy()

    # residuals and var
    # resid = Y - X_stack @ B  -> shape (N, T-p, n)
    resid = Y - np.matmul(X_stack, B)
    # compute var = resid.T @ resid / (T-p)
    Var_all = np.einsum('nti,ntj->nij', resid, resid) / (Y.shape[1])

    return A_all, Var_all


def vectorized_var_features(X_all, p=5, ridge=1e-6, verbose=True):
    """Compute VAR features (flattened A coeffs + logm(Var) upper-tri) for all samples."""
    N = X_all.shape[0]
    A_all, Var_all = vectorized_var_fit(X_all, p=p, ridge=ridge)

    # flatten A: each sample -> concatenated A matrices (p * n * n)
    n = A_all.shape[2]
    coeffs = A_all.reshape(N, -1)

    # compute logm of each siga
    var_vecs = []
    for i in range(N):
        try:
            Slog = sc.linalg.logm(Var_all[i])
        except Exception:
            # fallback: log of diag variances
            Slog = np.diag(np.log(np.diag(Var_all[i]) + 1e-12))
        iu = np.triu_indices(n)
        var_vecs.append(Slog[iu].real)
        if verbose and (i % 1000 == 0):
            print(f"logm processed {i}/{N}")
    var_vecs = np.vstack(var_vecs)

    feats = np.hstack([coeffs.real, var_vecs.real])
    return feats

if __name__ == "__main__": # create flag if loading does not work
    np_files = get_np_dir()
    x_train, y_train = load_train(np_files) # shape (7011,19,500)
    x_test, y_test = load_test(np_files) # shape (7011,19,500)

    # feature extraction
    # try to load labels from Excel documentation if available
    excel_path = PROJECT_ROOT / "data" / "Documentation" / "Seizures_Information.xlsx"
    if excel_path.exists():
        try:
            df = pd.read_excel(excel_path)
            # heuristics to find label column
            label_col = None
            for c in df.columns:
                if any(k in c.lower() for k in ("seiz", "label", "state")):
                    label_col = c
                    break
            if label_col is not None:
                if len(df) == len(x_train):
                    y_train = df[label_col].values
                    print(f"Loaded train labels from {excel_path} column '{label_col}'")
                elif len(df) == len(x_test):
                    y_test = df[label_col].values
                    print(f"Loaded test labels from {excel_path} column '{label_col}'")
                elif len(df) == len(x_train) + len(x_test):
                    y_train = df[label_col].values[: len(x_train)]
                    y_test = df[label_col].values[len(x_train) :]
                    print(f"Loaded combined labels from {excel_path} column '{label_col}'")
                else:
                    print(f"Found label column '{label_col}' but row count ({len(df)}) does not match train/test sizes; using .npy labels")
            else:
                print(f"No obvious label column found in {excel_path}; using .npy labels")
        except Exception as e:
            print(f"Failed to read {excel_path}: {e}; using .npy labels")
    else:
        print(f"No excel labels at {excel_path}; using .npy labels")

    print("Extracting train features (vectorized)...")
    Xf_train = vectorized_var_features(x_train, p=5, verbose=True)
    print("Extracting test features (vectorized)...")
    Xf_test = vectorized_var_features(x_test, p=5, verbose=True)

 
    from sklearn.decomposition import PCA
    pca = PCA(n_components=200)  # tune this using elbow method later
    Xf_train_p = pca.fit_transform(Xf_train) # pca w standarization
    Xf_test_p = pca.transform(Xf_test)

    clf = LogisticRegression(max_iter=1000, multi_class='ovr') # 1vall log regression
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0) # 
    scores = cross_val_score(clf, Xf_train_p, y_train, cv=cv, scoring='f1_macro')
    print("CV macro-F1:", scores.mean(), scores.std())

    clf.fit(Xf_train_p, y_train)
    test_score = clf.score(Xf_test_p, y_test)
    print("Test score:", test_score)
    results_dir = PROJECT_ROOT / "results" # mkdir if none exists
    results_dir.mkdir(parents=True, exist_ok=True)

    # save model + PCA
    dump({"clf": clf, "pca": pca}, results_dir / "var_clf.joblib")

    # save metrics (CV mean/std and test score)
    import json
    metrics = {
        "cv_macro_f1_mean": float(scores.mean()),
        "cv_macro_f1_std": float(scores.std()),
        "test_score": float(test_score)
    }
    with open(results_dir / "metrics.json", "w") as mf:
        json.dump(metrics, mf, indent=2)

    print(f"Saved model and metrics to: {results_dir}")
