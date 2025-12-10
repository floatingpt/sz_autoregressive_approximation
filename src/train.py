# src/train.py
import numpy as np
import os
from pathlib import Path
import sys
from collections import Counter
import json
from joblib import dump
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
    LeaveOneGroupOut,
    GroupKFold,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.covariance import LedoitWolf
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.utils.validation import check_is_fitted
from sklearn.calibration import IsotonicRegression
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR

# time series utilities
from statsmodels.tsa.stattools import pacf


# ensure project root in sys.path

try:
    PROJECT_ROOT = Path(__file__).parent.parent
except NameError:
    PROJECT_ROOT = Path(os.getcwd())

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FEATURE_CACHE = RESULTS_DIR / "cached_features"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_CACHE.mkdir(parents=True, exist_ok=True)


from var_model import  fit_var_window, var_features_from_A_Var
from collections import Counter
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from data.load_npy_dat import get_np_dir, load_train, load_test

# local helpers - assumed present in your repo
from data.load_npy_dat import get_np_dir, load_train, load_test
# If you have var_model helpers, try to import them but code has fallbacks

from var_model import fit_var_window, var_features_from_A_Var


def pacf_yw(series, max_lag):
    """Univariate PACF using Yule-Walker (unbiased) for reuse in plotting."""
    try:
        return pacf(series, nlags=max_lag, method="ywunbiased")
    except Exception:
        return pacf(series, nlags=max_lag)


def vectorized_var_fit(X_all, p, ridge=1e-6): 
    N, n_channels, T = X_all.shape
    if T <= p:
        raise ValueError("Time dimension T must be > order p")

    # transpose to (N, T, n)
    data = np.transpose(X_all, (0, 2, 1))  # (N, T, n)

    # build lagged design (N, T-p, n*p)
    lagged = []
    for k in range(p):
        lagged.append(data[:, p - k - 1:T - k - 1, :])  # (N, T-p, n)
    X_stack = np.concatenate(lagged, axis=2)  # (N, T-p, n*p)
    Y = data[:, p:, :]  # (N, T-p, n)

    # normal equations per-sample
    Xt = np.transpose(X_stack, (0, 2, 1))  # (N, n*p, T-p)
    XtX = np.matmul(Xt, X_stack)  # (N, n*p, n*p)
    XtY = np.matmul(Xt, Y)  # (N, n*p, n)

    # regularize XtX 
    M = XtX.shape[1]
    I = np.eye(M)[None, ...]  # (1, M, M)
    XtX_reg = XtX + ridge * I

    # solve for B: shape (N, n*p, n)
    B = np.linalg.solve(XtX_reg, XtY)

    # reshape B -> A_all (N, p, n, n)
    n = n_channels
    try:
        B_resh = B.reshape(N, n, p, n).transpose(0, 2, 1, 3)
    except Exception as e:
        # fallback to per-sample reshape if shapes inconsistent
        B_resh = np.zeros((N, p, n, n))
        for i in range(N):
            Bi = B[i]
            B_resh[i] = Bi.reshape(n, p, n).transpose(1, 0, 2)
    A_all = B_resh.copy()

    # residuals and covariance
    resid = Y - np.matmul(X_stack, B)  # (N, T-p, n)
    Tminus = resid.shape[1]
    # sample covariance per-sample: (n x n), denominator T-p
    Var_all = np.einsum('nti,ntj->nij', resid, resid) / float(Tminus)

    return A_all, Var_all


def companion_matrix_from_coefs(A_list):

    p = len(A_list)
    n = A_list[0].shape[0]
    top = np.hstack([A_list[k] for k in range(p)])  # n x (n*p)
    if p == 1:
        comp = top
    else:
        # bottom block: [I_{n(p-1)} , 0]
        bottom = np.vstack([np.eye(n * (p - 1)), np.zeros((n * (p - 1), n))])
        comp = np.vstack([top, bottom])
    return comp


def spectral_radius_of_companion(A_list):
    comp = companion_matrix_from_coefs(A_list)
    eigs = np.linalg.eigvals(comp)
    return float(np.max(np.abs(eigs)))


def var_features_unsigned_largest_diff(A_list, Sigma):
    """
    Extract features based on unsigned largest differences of AR basis functions (OLS coefficients).
    Enhanced with cross-lag interactions and robust statistics.
    """
    p = len(A_list)
    n = A_list[0].shape[0]
    
    # Unsigned largest differences per lag
    max_diffs = []
    for k in range(p):
        A_k = np.abs(A_list[k])
        max_val = np.max(A_k)
        min_val = np.min(A_k)
        diff = max_val - min_val
        max_diffs.append(diff)
    
    max_diffs = np.array(max_diffs, dtype=float)
    
    # Summary statistics across lags
    mean_diff = np.mean(max_diffs)
    std_diff = np.std(max_diffs) if len(max_diffs) > 1 else 0.0
    max_diff_overall = np.max(max_diffs)
    median_diff = np.median(max_diffs)
    
    # Cross-lag interaction features: measure temporal evolution
    lag_ratio = []
    for k in range(p - 1):
        if max_diffs[k] > 1e-8:
            ratio = max_diffs[k + 1] / (max_diffs[k] + 1e-8)
            lag_ratio.append(ratio)
    lag_ratio = np.array(lag_ratio, dtype=float) if lag_ratio else np.array([0.0])
    
    # Spectral norm (Frobenius) of aggregated coefficients
    A_agg = np.sum(np.abs(np.array([A_list[k] for k in range(p)])), axis=0)
    spectral_norm = np.linalg.norm(A_agg, ord='fro')
    
    # Covariance structure features
    try:
        eigvals = np.linalg.eigvalsh(Sigma)
        eigvals_sorted = np.sort(eigvals)[-min(n, 3):]  # top 3 eigenvalues
        cond_number = eigvals_sorted[-1] / (eigvals_sorted[0] + 1e-8) if len(eigvals_sorted) > 1 else 1.0
    except Exception:
        eigvals_sorted = np.zeros(min(n, 3), dtype=float)
        cond_number = 1.0
    
    feat = np.concatenate([
        max_diffs,  # per-lag max differences
        np.array([mean_diff, std_diff, max_diff_overall, median_diff], dtype=float),
        np.mean(lag_ratio, keepdims=True),  # temporal evolution
        np.array([spectral_norm, cond_number], dtype=float),
        eigvals_sorted
    ])
    
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    # Clip outliers to improve robustness
    feat = np.clip(feat, -100, 100)
    return feat


def var_features_from_A_Var_extended(A_list, Sigma):
    """Legacy extended features for backward compatibility."""
    return var_features_unsigned_largest_diff(A_list, Sigma)


# -------------------------
# Feature extraction (vectorized, cached, robust)
# -------------------------
def vectorized_var_features(X_all, p=5, ridge=1e-6, use_shrinkage=True, n_jobs=1, verbose=True, cache_path=None):
    """
    Compute VAR features for all samples. Will try to use fit_var_window() if available,
    otherwise fall back to local least-squares per-sample.
    Returns: feats: (N, D)
    Caches to cache_path if provided.
    """
    N = X_all.shape[0]
    if cache_path is not None and cache_path.exists():
        try:
            feats = np.load(cache_path)
            if verbose:
                print(f"Loaded cached features from {cache_path}")
            return feats
        except Exception:
            pass

    var_vecs = []

    # iterate (could be parallelized)
    for i in range(N):
        Xi = X_all[i]  # (channels, T)
        try:
            # try to use existing util (fit_var_window) if it exists and returns A_list, Sigma, ...
            A_list, Sigma, _ = fit_var_window(Xi, p)  # expects channels x T or channels x time?
        except Exception:
            # fallback to our per-sample LS
            data = Xi.T  # (T, n)
            if data.shape[0] <= p:
                print("Warning! Electrode count less than order")
                # insufficient length for given order; produce zero feature vector (length unknown)
                # produce a small zero vector of expected size: guess dims (p + p + n + n + 2 + min(n,6))
                n_ch = data.shape[1]
                guess_len = p + p + n_ch + n_ch + 2 + min(n_ch, 6)
                var_vecs.append(np.zeros((guess_len,), dtype=float))
                continue

            # design matrix
            # build design with shape (T-p, n*p)
            design_cols = [data[p - k - 1:-k - 1, :] for k in range(p)]
            design = np.hstack(design_cols)  # (T-p, n*p)
            Y = data[p:]  # (T-p, n)
            # solve B: (n*p, n)
            B, *_ = np.linalg.lstsq(design, Y, rcond=None)
            # coefs -> list of A (p, n, n)
            n = data.shape[1]
            coefs = B.T.reshape(n, p, n).transpose(1, 0, 2)
            A_list = [coefs[k] for k in range(p)]
            resid = Y - design.dot(B)
            # shrink covariance
            if use_shrinkage:
                try:
                    lw = LedoitWolf().fit(resid)
                    Sigma = lw.covariance_
                except Exception:
                    Sigma = np.cov(resid.T)
            else:
                Sigma = np.cov(resid.T)

        # features from A_list and Sigma
        feat_i = None
        try:
            # If var_features_from_A_Var exists externally, prefer it and augment with extended features
            if var_features_from_A_Var is not None:
                feat_base = var_features_from_A_Var(A_list, Sigma)
                feat_ext = var_features_from_A_Var_extended(A_list, Sigma)
                feat_i = np.concatenate([np.atleast_1d(feat_base).ravel(), feat_ext.ravel()])
            else:
                feat_i = var_features_from_A_Var_extended(A_list, Sigma)
        except Exception:
            feat_i = var_features_from_A_Var_extended(A_list, Sigma)

        var_vecs.append(feat_i)
        if verbose and (i % 500 == 0):
            print(f"Processed {i}/{N} samples")

    var_vecs = np.vstack(var_vecs)
    feats = np.nan_to_num(var_vecs.real, nan=0.0, posinf=0.0, neginf=0.0)
    if cache_path is not None:
        np.save(cache_path, feats)
    return feats


#pacf order electioj
def select_global_var_order_pacf_statsmodels(X_all, p_max=12, sample_limit=200, alpha=0.05, verbose=True):
    
    n_samples = min(len(X_all), sample_limit)
    selected_orders = []
    z = 1.96  # rough two-sided
    for i in range(n_samples):
        X = X_all[i]  # (channels, T)
        if X.shape[1] < 10:
            continue
        T = X.shape[1]
        thresh = z / np.sqrt(T)
        per_channel_p = []
        for ch in range(X.shape[0]):
            try:
                pacf_vals = pacf(X[ch], nlags=p_max, method='ywunbiased')
            except Exception:
                continue
            sig_lags = [lag for lag in range(1, len(pacf_vals)) if abs(pacf_vals[lag]) > thresh]
            per_channel_p.append(max(sig_lags) if len(sig_lags) else 1)
        if per_channel_p:
            p_i = int(max(per_channel_p))
            selected_orders.append(p_i)
    if len(selected_orders) == 0:
        if verbose:
            print("No PACF-based orders found; defaulting to p=5")
        return 5
    mode_p = Counter(selected_orders).most_common(1)[0][0]
    if verbose:
        print(f"PACF-selected modal p across subset: {mode_p}")
    return mode_p


# fits order p model then uses Frobenius norm on each lag coefficient matrix for lag coefficient. Largest difference is used as mode across windows
def select_global_var_order_mpacf(X_all, p_max=12, sample_limit=200, alpha=0.05, verbose=True):

    n_samples = min(len(X_all), sample_limit)
    if n_samples == 0:
        return 5
    rng = np.random.default_rng(0)
    chosen = []
    z = 1.96 # significance 0.05
    for idx in rng.choice(len(X_all), size=n_samples, replace=False):
        X = X_all[idx]  # (channels, T)
        T = X.shape[1]
        if T <= p_max + 2 or X.shape[0] < 2:
            continue
        try:
            model = VAR(X.T)
            # trend='n' to avoid intercept effects on small windows
            res = model.fit(maxlags=p_max, trend='n')
            coefs = res.coefs  # shape (p, k, k)
            if coefs.size == 0:
                continue
            norms = [np.linalg.norm(coefs[k], ord='fro') for k in range(coefs.shape[0])]
            thresh = z / np.sqrt(T)
            sig_lags = [k + 1 for k, nrm in enumerate(norms) if nrm > thresh]
            p_i = max(sig_lags) if sig_lags else 1
            chosen.append(p_i)
        except Exception:
            continue
    if not chosen:
        if verbose:
            print("mPACF selection failed; defaulting to p=5")
        return 5
    mode_p = Counter(chosen).most_common(1)[0][0]
    if verbose:
        print(f"mPACF-selected modal p across subset: {mode_p}")
    return mode_p


def select_global_var_order_bic_subset(X_all, p_max=12, n_samples=50, verbose=True):
    if len(X_all) == 0:
        return 5
    rng = np.random.default_rng(0)
    indices = rng.choice(len(X_all), size=min(len(X_all), n_samples), replace=False)
    bic_accum = {p: [] for p in range(1, p_max + 1)}
    for idx in indices:
        X = X_all[idx]
        if X.shape[1] <= p_max + 2:
            continue
        try:
            model = VAR(X.T)
            for p in range(1, p_max + 1):
                try:
                    res = model.fit(maxlags=p, trend='n')
                    bic_accum[p].append(res.bic)
                except Exception:
                    continue
        except Exception:
            continue
    # compute mean BIC per p
    mean_bic = {p: np.mean(vals) for p, vals in bic_accum.items() if len(vals) > 0}
    if not mean_bic:
        if verbose:
            print("BIC selection failed; defaulting to p=5")
        return 5
    best_p = min(mean_bic, key=mean_bic.get)
    if verbose:
        print(f"BIC-selected p (mean across subset): {best_p}")
    return best_p


# uses pacf, mpacf, and BIC to return most common p order
def select_global_var_order_ensemble(X_all, p_max=12, verbose=True):

    methods = [
        ("PACF", select_global_var_order_pacf_statsmodels(X_all, p_max, verbose=False)),
        ("mPACF", select_global_var_order_mpacf(X_all, p_max, verbose=False)),
        ("BIC", select_global_var_order_bic_subset(X_all, p_max, verbose=False))
    ]
    
    orders = [p for _, p in methods]
    mode_order = Counter(orders).most_common(1)[0][0]
    
    if verbose:
        order_str = ", ".join([f"{name}: {p}" for name, p in methods])
        print(f"Ensemble order selection: [{order_str}] → mode: {mode_order}")
    
    return mode_order


# -------------------------
# Main script
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAR classifier with order selection")
    parser.add_argument("--order-method", choices=["pacf", "mpacf", "bic", "ensemble"], default="pacf",
                        help="Method to choose VAR order: pacf (per-channel), mpacf (multivariate), bic (subset BIC), ensemble (voting)")
    parser.add_argument("--p-max", type=int, default=12, help="Maximum order to consider")
    parser.add_argument("--bic-samples", type=int, default=50, help="Number of windows for BIC subset evaluation")
    args = parser.parse_args()

    np_files = get_np_dir()
    x_train, y_train = load_train(np_files)  # shape (N, channels, T)
    x_test, y_test = load_test(np_files)

    # difference to capture changes (beware this shortens T by 1)
    x_train_diff = np.diff(x_train, axis=2)
    x_test_diff = np.diff(x_test, axis=2)

    y_train_bin = (y_train != 0).astype(int)
    y_test_bin = (y_test != 0).astype(int)

    print(f"Labels train: {np.bincount(y_train_bin.flatten())}   test: {np.bincount(y_test_bin.flatten())}")

    # Try to load patient IDs for group-aware CV
    patient_ids_train_path = np_files / "patient_ids_train.npy"
    patient_ids_test_path = np_files / "patient_ids_test.npy"
    groups_train = None
    groups_test = None
    if patient_ids_train_path.exists() and patient_ids_test_path.exists():
        try:
            groups_train = np.load(patient_ids_train_path)
            groups_test = np.load(patient_ids_test_path)
            if len(groups_train) != len(x_train) or len(groups_test) != len(x_test):
                print("Warning: patient ID lengths don't match data; ignoring groups")
                groups_train = None
                groups_test = None
            else:
                print(f"Loaded patient groups: {len(np.unique(groups_train))} train patients, {len(np.unique(groups_test))} test patients")
        except Exception as e:
            print(f"Could not load patient ids: {e}. Continuing without groups.")
            groups_train = None
            groups_test = None
    else:
        print("Patient IDs not found; running non-group CV (may leak across patients).")

    # Choose global VAR order
    if args.order_method == "pacf":
        print("Estimating global VAR order using per-channel PACF on differenced series...")
        p = select_global_var_order_pacf_statsmodels(x_train_diff, p_max=args.p_max, sample_limit=300)
    elif args.order_method == "mpacf":
        print("Estimating global VAR order using multivariate PACF (VAR-based) on differenced series...")
        p = select_global_var_order_mpacf(x_train_diff, p_max=args.p_max, sample_limit=300)
    elif args.order_method == "bic":
        print(f"Estimating global VAR order using BIC across a subset of {args.bic_samples} windows...")
        p = select_global_var_order_bic_subset(x_train_diff, p_max=args.p_max, n_samples=args.bic_samples)
    else:  # ensemble
        print("Estimating global VAR order using ensemble voting (PACF, mPACF, BIC)...")
        p = select_global_var_order_ensemble(x_train_diff, p_max=args.p_max)
    print(f"Selected global VAR order p={p}")

    # feature extraction with caching
    train_cache = FEATURE_CACHE / f"var_feats_train_p{p}.npy"
    test_cache = FEATURE_CACHE / f"var_feats_test_p{p}.npy"

    print("Extracting train features (vectorized; cached)...")
    Xf_train = vectorized_var_features(x_train_diff, p=p, cache_path=train_cache, verbose=True)
    print("Extracting test features (vectorized; cached)...")
    Xf_test = vectorized_var_features(x_test_diff, p=p, cache_path=test_cache, verbose=True)

    # Standardize features and (optionally) PCA inside a pipeline
    scaler = StandardScaler()
    # Determine PCA components automatically with cap
    use_pca = True
    n_components_optimal = None
    if use_pca:
        # quick heuristic: pick components that explain 95% var up to cap 40
        pca_temp = PCA(n_components=min(40, Xf_train.shape[1]))
        pca_temp.fit(scaler.fit_transform(Xf_train))
        cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components_optimal = int(np.searchsorted(cumsum, 0.95) + 1)
        n_components_optimal = max(2, min(n_components_optimal, 40))
        print(f"Auto-selected PCA components: {n_components_optimal}")
    else:
        n_components_optimal = None

    # Label encoding
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_bin.ravel())
    y_test_enc = le.transform(y_test_bin.ravel())

    # Build pipeline
    steps = [("scaler", StandardScaler())]
    if use_pca and n_components_optimal is not None:
        steps.append(("pca", PCA(n_components=n_components_optimal)))
    steps.append(("clf", LogisticRegression(max_iter=2000, class_weight=None, random_state=0)))
    pipeline = Pipeline(steps)

    # Grid for logistic regression C and class_weight variants
    param_grid = {
        "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__class_weight": [None, "balanced"]
    }

    # Inner CV for hyperparameter tuning (group aware if groups exist)
    if groups_train is not None:
        unique_groups = np.unique(groups_train)
        n_groups = len(unique_groups)
        inner_cv_splits = min(5, n_groups) if n_groups >= 2 else 2
        inner_cv = GroupKFold(n_splits=inner_cv_splits)
        use_groups_for_grid = True
        print(f"Using GroupKFold(n_splits={inner_cv_splits}) for inner CV during grid search.")
    else:
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        use_groups_for_grid = False
        print("Using StratifiedKFold for inner CV during grid search.")

    grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring="f1_macro", n_jobs=-1, verbose=2)
    if use_groups_for_grid:
        # GridSearchCV doesn't accept groups for internal splitting; we must pass groups via .split when fitting
        # But sklearn's GridSearchCV.fit accepts groups argument and will forward to cv.split if cv is group-aware.
        grid_search.fit(Xf_train, y_train_enc, groups=groups_train)
    else:
        grid_search.fit(Xf_train, y_train_enc)

    #print(f"Grid search best params: {grid_search.best_params_}")
    #print(f"Grid search best CV f1_macro: {grid_search.best_score_:.4f}")

    best_pipeline = grid_search.best_estimator_

    # Outer CV for final evaluation: LeaveOneGroupOut if groups available, else StratifiedKFold
    if groups_train is not None:
        print("Final evaluation using Leave-One-Group-Out (patient-level).")
        logo = LeaveOneGroupOut()
        cv_splits = logo.split(Xf_train, y_train_enc, groups_train)
        scores_f1 = []
        scores_auc = []
        scores_ap = []
        for train_idx, val_idx in cv_splits:
            Xtr, Xval = Xf_train[train_idx], Xf_train[val_idx]
            ytr, yval = y_train_enc[train_idx], y_train_enc[val_idx]
            # retrain best_pipeline on train fold
            best_pipeline.fit(Xtr, ytr)
            ypred = best_pipeline.predict(Xval)
            yprob = best_pipeline.predict_proba(Xval)[:, 1] if hasattr(best_pipeline, "predict_proba") else None
            scores_f1.append(f1_score(yval, ypred, average="macro"))
            if yprob is not None:
                try:
                    scores_auc.append(roc_auc_score(yval, yprob))
                    scores_ap.append(average_precision_score(yval, yprob))
                except Exception:
                    scores_auc.append(np.nan)
                    scores_ap.append(np.nan)
        scores_f1 = np.array(scores_f1)
        #print(f"Outer CV F1-macro: {np.nanmean(scores_f1):.4f} ± {np.nanstd(scores_f1):.4f}")
    else:
        #print("Final evaluation using StratifiedKFold.")
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        scores = cross_val_score(best_pipeline, Xf_train, y_train_enc, cv=outer_cv, scoring="f1_macro", n_jobs=-1)
        #print(f"Outer CV F1-macro: {scores.mean():.4f} ± {scores.std():.4f}")

    # Fit final model on full training set and evaluate on test set
    best_pipeline.fit(Xf_train, y_train_enc)
    y_pred_test = best_pipeline.predict(Xf_test)
    #train_acc = best_pipeline.score(Xf_train, y_train_enc)
    #test_acc = best_pipeline.score(Xf_test, y_test_enc)
    test_f1 = f1_score(y_test_enc, y_pred_test, average="macro")
    y_prob_test = best_pipeline.predict_proba(Xf_test)[:, 1] if hasattr(best_pipeline, "predict_proba") else None
    test_auc = roc_auc_score(y_test_enc, y_prob_test) if y_prob_test is not None else np.nan
    test_ap = average_precision_score(y_test_enc, y_prob_test) if y_prob_test is not None else np.nan

    #print(f"Training Accuracy: {train_acc:.4f}")
    #print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test F1-macro: {test_f1:.4f}")
    print(f"Test AUC: {test_auc:.4f}   AP: {test_ap:.4f}")
    
    # Threshold-based binary classification using AUC (via ROC curve optimal threshold)
    optimal_threshold = 0.5  # default
    if y_prob_test is not None and not np.isnan(test_auc):
        # Find optimal threshold using Youden's J statistic
        fpr, tpr, thresholds = roc_curve(y_test_enc, y_prob_test)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Apply threshold-based binary classification
        y_pred_threshold = (y_prob_test >= optimal_threshold).astype(int)
        test_f1_threshold = f1_score(y_test_enc, y_pred_threshold, average="macro", zero_division=0)
        test_acc_threshold = np.mean(y_pred_threshold == y_test_enc)
        
        #print(f"\nThreshold-based classification (optimal threshold: {optimal_threshold:.4f}):")
        #print(f"  Test Accuracy (threshold): {test_acc_threshold:.4f}")
        #print(f"  Test F1-macro (threshold): {test_f1_threshold:.4f}")
    else:
        test_f1_threshold = np.nan
        test_acc_threshold = np.nan
    
    # Collect outer CV scores if available
    outer_cv_scores = {}
    if groups_train is not None:
        outer_cv_scores = {
            "f1_macro": scores_f1.tolist() if 'scores_f1' in locals() else [],
            "auc": scores_auc if 'scores_auc' in locals() else [],
            "ap": scores_ap if 'scores_ap' in locals() else [],
        }
    else:
        outer_cv_scores = {
            "f1_macro": scores.tolist() if 'scores' in locals() else [],
            "auc": [],
            "ap": [],
        }

    # Save artifacts
    artifacts = {
        "pipeline": best_pipeline,
        "label_encoder": le,
        "p": int(p),
        "n_pca_components": int(n_components_optimal) if n_components_optimal is not None else None,
        "grid_search_cv_best": grid_search.best_params_,
        "optimal_threshold": float(optimal_threshold),
    }
    dump(artifacts, RESULTS_DIR / "var_clf_pipeline.joblib")
    
    # Comprehensive metrics dictionary
    metrics = {
        "order_method": args.order_method,
        "feature_type": "unsigned_largest_differences_ols",
        "p": int(p),
        "n_pca_components": int(n_components_optimal) if n_components_optimal is not None else None,
        "best_params": grid_search.best_params_,
        "best_cv_f1_macro": float(grid_search.best_score_),
        #"test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
        "test_auc": float(test_auc) if not np.isnan(test_auc) else None,
        "test_average_precision": float(test_ap) if not np.isnan(test_ap) else None,
        "optimal_threshold": float(optimal_threshold),
        "test_accuracy_threshold": float(test_acc_threshold) if not np.isnan(test_acc_threshold) else None,
        "test_f1_macro_threshold": float(test_f1_threshold) if not np.isnan(test_f1_threshold) else None,
        "outer_cv_scores": outer_cv_scores,
    }
    
    # Save metrics to JSON
    with open(RESULTS_DIR / "metrics.json", "w") as mf:
        json.dump(metrics, mf, indent=2)
    
    # Create plots for AUC and F1 macro
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: F1 Macro comparison
    f1_scores_data = {
        "Best CV F1": grid_search.best_score_,
        "Test F1 (prob)": test_f1,
        "Test F1 (threshold)": test_f1_threshold,
    }
    if outer_cv_scores["f1_macro"]:
        f1_scores_data["Outer CV F1 (mean)"] = np.mean(outer_cv_scores["f1_macro"])
    
    ax1 = axes[0]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars1 = ax1.bar(range(len(f1_scores_data)), list(f1_scores_data.values()), color=colors[:len(f1_scores_data)])
    ax1.set_ylabel('F1 Macro Score', fontsize=12, fontweight='bold')
    ax1.set_title('F1 Macro Scores Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(f1_scores_data)))
    ax1.set_xticklabels(list(f1_scores_data.keys()), rotation=15, ha='right')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: AUC comparison
    auc_scores_data = {
        "Test AUC": test_auc,
    }
    if outer_cv_scores["auc"]:
        auc_mean = np.mean([s for s in outer_cv_scores["auc"] if not np.isnan(s)])
        auc_scores_data["Outer CV AUC (mean)"] = auc_mean
    
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(auc_scores_data)), list(auc_scores_data.values()), 
                     color=colors[:len(auc_scores_data)])
    ax2.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax2.set_title('AUC Scores Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(auc_scores_data)))
    ax2.set_xticklabels(list(auc_scores_data.keys()), rotation=15, ha='right')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        if not np.isnan(height):
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plot_path = FIGURES_DIR / "auc_f1_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {plot_path}")
    plt.close()
    
    # Create detailed metrics plot if outer CV scores are available
    if outer_cv_scores["f1_macro"] and len(outer_cv_scores["f1_macro"]) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot outer CV F1 macro distribution
        ax1 = axes[0]
        ax1.hist(outer_cv_scores["f1_macro"], bins=10, alpha=0.7, color='#3498db', edgecolor='black')
        ax1.axvline(np.mean(outer_cv_scores["f1_macro"]), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(outer_cv_scores["f1_macro"]):.4f}')
        ax1.set_xlabel('F1 Macro Score', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Distribution of Outer CV F1 Macro Scores', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Plot outer CV AUC distribution if available
        ax2 = axes[1]
        if outer_cv_scores["auc"] and len([s for s in outer_cv_scores["auc"] if not np.isnan(s)]) > 0:
            auc_valid = [s for s in outer_cv_scores["auc"] if not np.isnan(s)]
            ax2.hist(auc_valid, bins=10, alpha=0.7, color='#e74c3c', edgecolor='black')
            ax2.axvline(np.mean(auc_valid), color='darkred', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(auc_valid):.4f}')
            ax2.set_xlabel('AUC Score', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax2.set_title('Distribution of Outer CV AUC Scores', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
        else:
            ax2.text(0.5, 0.5, 'No AUC scores available', ha='center', va='center', fontsize=12)
            ax2.axis('off')
        
        plt.tight_layout()
        dist_plot_path = FIGURES_DIR / "cv_scores_distribution.png"
        plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved distribution plot to {dist_plot_path}")
        plt.close()
    
    # Create ROC and Precision-Recall curves for test set
    if y_prob_test is not None and not np.isnan(test_auc):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # ROC Curve
        ax1 = axes[0]
        fpr, tpr, _ = roc_curve(y_test_enc, y_prob_test)
        ax1.plot(fpr, tpr, color='#2c3e50', lw=2.5, label=f'ROC Curve (AUC = {test_auc:.4f})')
        ax1.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Classifier')
        ax1.fill_between(fpr, tpr, alpha=0.2, color='#3498db')
        ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax1.set_title('ROC Curve (Test Set)', fontsize=13, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=11)
        ax1.grid(alpha=0.3, linestyle='--')
        ax1.set_xlim([-0.02, 1.02])
        ax1.set_ylim([-0.02, 1.02])
        
        # Precision-Recall Curve
        ax2 = axes[1]
        precision, recall, _ = precision_recall_curve(y_test_enc, y_prob_test)
        ax2.plot(recall, precision, color='#e74c3c', lw=2.5, label=f'PR Curve (AP = {test_ap:.4f})')
        ax2.axhline(y=np.mean(y_test_enc), color='gray', lw=1.5, linestyle='--', 
                   label=f'Baseline (Prevalence = {np.mean(y_test_enc):.4f})')
        ax2.fill_between(recall, precision, alpha=0.2, color='#e74c3c')
        ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax2.set_title('Precision-Recall Curve (Test Set)', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(alpha=0.3, linestyle='--')
        ax2.set_xlim([-0.02, 1.02])
        ax2.set_ylim([-0.02, 1.02])
        
        plt.tight_layout()
        roc_pr_path = FIGURES_DIR / "roc_pr_curves.png"
        plt.savefig(roc_pr_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC and Precision-Recall curves to {roc_pr_path}")
        plt.close()

    print(f"Saved pipeline and metrics to {RESULTS_DIR}")
