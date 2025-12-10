#Generates ROC, Precision-Recall, Confusion Matrix, Calibration plots
# generates json of model accuracy
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


try:
    PROJECT_ROOT = Path(__file__).parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ensure src is importable so we can reuse vectorized_var_features
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.load_npy_dat import get_np_dir, load_test
from train import vectorized_var_features, select_global_var_order_pacf, autocorr, pacf_yw


def ensure_dirs():
    out = PROJECT_ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_roc(y_true, y_score, classes, out_path):
    n_classes = len(classes)
    y_bin = label_binarize(y_true, classes=classes)

    plt.figure(figsize=(8, 6))
    # per-class
    aucs = {}
    for i, cls in enumerate(classes):
        if n_classes == 2 and y_bin.shape[1] == 1:
            # binary label_binarize returns single column; handle indexing
            gt = y_bin.ravel()
            sc = y_score[:, 1] if y_score.shape[1] > 1 else y_score.ravel()
        else:
            gt = y_bin[:, i]
            sc = y_score[:, i]
        fpr, tpr, _ = roc_curve(gt, sc)
        roc_auc = auc(fpr, tpr)
        aucs[str(cls)] = float(roc_auc)
        plt.plot(fpr, tpr, lw=2, label=f"class {cls} (AUC = {roc_auc:.3f})")

    # micro-average: only when score shape matches binarized labels
    if y_bin.size > 0 and y_score.shape[1] == y_bin.shape[1]:
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)
        aucs["micro"] = float(roc_auc)
        plt.plot(fpr, tpr, color="navy", lw=2, linestyle="--", label=f"micro (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return aucs


def plot_pr(y_true, y_score, classes, out_path):
    y_bin = label_binarize(y_true, classes=classes)
    plt.figure(figsize=(8, 6))
    aps = {}
    for i, cls in enumerate(classes):
        if y_bin.shape[1] == 1:
            gt = y_bin.ravel()
            sc = y_score[:, 1] if y_score.shape[1] > 1 else y_score.ravel()
        else:
            gt = y_bin[:, i]
            sc = y_score[:, i]
        prec, rec, _ = precision_recall_curve(gt, sc)
        ap = average_precision_score(gt, sc)
        aps[str(cls)] = float(ap)
        plt.plot(rec, prec, lw=2, label=f"class {cls} (AP = {ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return aps


def plot_confusion(y_true, y_pred, classes, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return cm.tolist()


def plot_calibration(y_true, y_score, classes, out_path, n_bins=10):
    y_bin = label_binarize(y_true, classes=classes)
    plt.figure(figsize=(8, 6))
    calib_stats = {}
    for i, cls in enumerate(classes):
        if y_bin.shape[1] == 1:
            gt = y_bin.ravel()
            sc = y_score[:, 1] if y_score.shape[1] > 1 else y_score.ravel()
        else:
            gt = y_bin[:, i]
            sc = y_score[:, i]
        prob_true, prob_pred = calibration_curve(gt, sc, n_bins=n_bins, strategy='uniform')
        calib_stats[str(cls)] = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }
        plt.plot(prob_pred, prob_true, marker='o', label=f'class {cls}')

    plt.plot([0, 1], [0, 1], 'k:', label='perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return calib_stats

def plot_pacf_distribution(p_values, out_path):
    """Plot distribution of PACF-selected VAR orders"""
    plt.figure(figsize=(6, 4))
    plt.hist(p_values, bins=np.arange(1, max(p_values) + 2) - 0.5, edgecolor='black', alpha=0.7)
    plt.xlabel("Selected VAR order p")
    plt.ylabel("Count")
    plt.title("Distribution of PACF-selected VAR orders")
    plt.xticks(np.arange(1, max(p_values) + 1))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def compute_pacf_orders_test(x_test, p_max=12):
    """Compute PACF-based VAR orders for each test sample"""
    from collections import Counter
    p_vals = []
    z = 1.96  # 95% confidence
    
    for i, X in enumerate(x_test):
        if X.shape[1] < 10:
            p_vals.append(1)
            continue
        
        T = X.shape[1]
        thresh = z / np.sqrt(T)
        per_channel_p = []
        
        for ch in range(X.shape[0]):
            series = X[ch]
            pacf_vals = pacf_yw(series, p_max)
            sig_lags = [lag for lag in range(1, len(pacf_vals)) if abs(pacf_vals[lag]) > thresh]
            
            if len(sig_lags) == 0:
                per_channel_p.append(1)
            else:
                per_channel_p.append(min(max(sig_lags), p_max))
        
        if len(per_channel_p) > 0:
            p_i = int(max(per_channel_p))
            p_vals.append(p_i)
        else:
            p_vals.append(1)
    
    return p_vals




def coef_summary(clf, out_path):
    # For logistic regression, coef_ shape (n_classes, n_features) or (1, n_features)
    coefs = getattr(clf, 'coef_', None)
    if coefs is None:
        return None
    plt.figure(figsize=(8, 4))
    plt.boxplot(np.abs(coefs).T)
    plt.title('Distribution of absolute classifier coefficients')
    plt.ylabel('abs(coef)')
    plt.xlabel('feature index')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def main():
    out_dir = ensure_dirs()
    model_path = PROJECT_ROOT / 'results' / 'var_clf.joblib'
    if not model_path.exists():
        print(f"Model file not found: {model_path}. Run training first.")
        return

    model = load(model_path)
    clf = model.get('clf', model if hasattr(model, 'predict') else None)
    pca = model.get('pca', None)
    le = model.get('le', None)
    if clf is None:
        print('Failed to locate classifier in joblib file')
        return

    np_files = get_np_dir()
    x_test, y_test = load_test(np_files)

    # Use differenced series and binary seizure labels to match training pipeline
    x_test_diff = np.diff(x_test, axis=2)
    y_test_bin = (y_test != 0).astype(int)

    # Decide which VAR order `p` to use for feature extraction.
    # Prefer an explicit saved `p` in the model, otherwise try to infer it from the saved PCA input size.
    saved_p = model.get('p', None)
    inferred_p = None
    n_ch = x_test.shape[1]
    tri_len = n_ch * (n_ch + 1) // 2
    if pca is not None:
        n_in = getattr(pca, 'n_features_in_', None)
        if n_in is not None:
            # solve for p in: n_in = p*n_ch*n_ch + tri_len
            residual = n_in - tri_len
            if residual > 0 and residual % (n_ch * n_ch) == 0:
                inferred_p = residual // (n_ch * n_ch)
    if saved_p is not None:
        p_used = saved_p
        print(f"Using saved VAR order p={p_used} from model file")
    elif inferred_p is not None:
        p_used = inferred_p
        print(f"Inferred VAR order p={p_used} from PCA input dimension")
    else:
        p_used = 5
        print(f"No saved/inferred VAR order found; defaulting to p={p_used}")

    print(f'Computing VAR features for test set with p={p_used} (differenced series)...')
    Xf_test = vectorized_var_features(x_test_diff, p=p_used, verbose=True)
    if pca is not None:
        # defensive check: ensure PCA was fitted on same number of features
        n_in = getattr(pca, "n_features_in_", None)
        if n_in is not None and n_in != Xf_test.shape[1]:
            msg = (
                f"PCA feature mismatch: saved PCA was fitted with {n_in} features, "
                f"but current test features have {Xf_test.shape[1]} columns.\n"
                "Likely cause: the model (and PCA) was trained with a different feature pipeline.\n"
                "Recommended fix: re-run the training script `min_viable_ar_approx/src/train.py` so the saved PCA matches the current feature extractor.\n"
            )
            print(msg)
            allow_refit = True
            if allow_refit:
                print("Warning: refitting PCA on test data (this causes data leakage). Re-fitting because allow_refit=True.")
                from sklearn.decomposition import PCA as PCAnew
                pca_new = PCAnew(n_components=min(getattr(pca, 'n_components_', getattr(pca, 'n_components', 2)), Xf_test.shape[1]))
                Xf_test_p = pca_new.fit_transform(Xf_test)
                pca = pca_new
            else:
                raise RuntimeError(msg + "If you understand the consequences and still want to proceed, set allow_refit=True in this script to auto-refit (not recommended).")
        else:
            Xf_test_p = pca.transform(Xf_test)
    else:
        Xf_test_p = Xf_test

    # probabilities and preds
    try:
        y_score = clf.predict_proba(Xf_test_p)
    except Exception:
        # fallback: decision_function 
        df = clf.decision_function(Xf_test_p)
        # softmax
        exp = np.exp(df - np.max(df, axis=1, keepdims=True))
        y_score = exp / exp.sum(axis=1, keepdims=True)

    y_pred = clf.predict(Xf_test_p)

    # Diagnostics: how many positives are being predicted and score range
    if hasattr(clf, "classes_") and len(clf.classes_) == 2:
        pos_idx = list(clf.classes_).index(1)
        pos_scores = y_score[:, pos_idx]
        print(f"Prediction summary: predicted positives = {(y_pred==1).sum()} / {len(y_pred)}")
        print(f"Positive class score stats: min={pos_scores.min():.4f}, max={pos_scores.max():.4f}, mean={pos_scores.mean():.4f}")
    else:
        print(f"Prediction summary: class counts = {np.bincount(y_pred.astype(int))}")

    # Encode / map test labels to the training label space (binary)
    if le is not None:
        try:
            y_test_enc = le.transform(y_test_bin)
        except Exception:
            # Best-effort mapping: if unique test labels count matches classes, map by order
            uniq = np.unique(y_test_bin)
            if len(uniq) == len(clf.classes_):
                mapping = {orig: enc for orig, enc in zip(uniq, clf.classes_)}
                y_test_enc = np.array([mapping[v] for v in y_test_bin])
            else:
                try:
                    y_test_enc = y_test_bin.astype(int)
                except Exception:
                    raise RuntimeError('Could not map test labels to training label encoding. Retrain saving the LabelEncoder or provide matching labels.')
    else:
        y_test_enc = y_test_bin

    # Ensure both binary classes are present for metrics shapes
    if hasattr(clf, 'classes_') and len(clf.classes_) == 2:
        classes = np.array(clf.classes_)
    else:
        classes = np.unique(np.concatenate([y_test_enc, y_pred]))

    results = {}
    
    # Compute PACF-based VAR orders for test set (matches training approach)
    print('Computing PACF-selected VAR orders for test samples...')
    p_vals = compute_pacf_orders_test(x_test_diff, p_max=12)
    results['pacf_selected_orders'] = p_vals
    
    # Plot distribution
    plot_pacf_distribution(p_vals, out_dir / 'pacf_distribution.png')
    print(f'  PACF order distribution: min={min(p_vals)}, max={max(p_vals)}, '
          f'mean={np.mean(p_vals):.2f}, median={np.median(p_vals):.0f}')

    print('Plotting ROC...')
    results['roc_aucs'] = plot_roc(y_test_enc, y_score, classes, out_dir / 'roc.png')

    print('Plotting Precision-Recall...')
    results['pr_aps'] = plot_pr(y_test_enc, y_score, classes, out_dir / 'pr.png')

    print('Plotting Confusion Matrix...')
    results['confusion_matrix'] = plot_confusion(y_test_enc, y_pred, classes, out_dir / 'confusion.png')

    print('Plotting Calibration Curve...')
    results['calibration'] = plot_calibration(y_test_enc, y_score, classes, out_dir / 'calibration.png')

    print('Plotting Coefficient Summary...')
    coef_summary(clf, out_dir / 'coeffs.png')
    report = classification_report(y_test_enc, y_pred, output_dict=True, zero_division=0)
    results['classification_report'] = report

    # write summary
    with open(PROJECT_ROOT / 'results' / 'metrics_plots.json', 'w') as fh:
        json.dump(results, fh, indent=2)

    print(f"Saved diagnostic figures to: {out_dir}")


if __name__ == '__main__':
    main()
