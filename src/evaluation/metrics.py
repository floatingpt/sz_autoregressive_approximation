"""
Comprehensive evaluation metrics for seizure classification.

Provides clinically relevant metrics beyond simple accuracy:
- ROC-AUC
- Precision, Recall, F1
- Sensitivity (recall for seizure class)
- Per-subject aggregated metrics
"""

from __future__ import annotations

from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Parameters
    ----------
    y_true : ndarray, shape (n_samples,)
        True binary labels
    y_pred : ndarray, shape (n_samples,)
        Predicted binary labels
    y_proba : ndarray, shape (n_samples, 2) or (n_samples,), optional
        Predicted probabilities for positive class
        
    Returns
    -------
    metrics : dict
        Dictionary containing:
        - accuracy: Overall accuracy
        - precision_seizure: Precision for seizure class
        - recall_seizure: Recall (sensitivity) for seizure class
        - f1_seizure: F1 score for seizure class
        - precision_nonseizure: Precision for non-seizure class
        - recall_nonseizure: Recall (specificity) for non-seizure class
        - f1_nonseizure: F1 score for non-seizure class
        - roc_auc: ROC-AUC score (if probabilities provided)
        - pr_auc:  Precision-Recall AUC (Average Precision) if probabilities provided
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    
    # Non-seizure class (class 0)
    metrics['precision_nonseizure'] = precision[0]
    metrics['recall_nonseizure'] = recall[0]  # Specificity
    metrics['f1_nonseizure'] = f1[0]
    
    # Seizure class (class 1)
    metrics['precision_seizure'] = precision[1]
    metrics['recall_seizure'] = recall[1]  # Sensitivity
    metrics['f1_seizure'] = f1[1]
    
    # Sensitivity and specificity (clinical terms)
    metrics['sensitivity'] = recall[1]  # True positive rate for seizures
    metrics['specificity'] = recall[0]  # True negative rate for non-seizures
    
    # ROC-AUC if probabilities provided
    if y_proba is not None:
        # Handle both (n, 2) and (n,) shapes
        if y_proba.ndim == 2:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba
            
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba_pos)
        except ValueError:
            metrics['roc_auc'] = np.nan

        # Average Precision (PR-AUC)
        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_proba_pos)
        except ValueError:
            metrics['pr_auc'] = np.nan
            
    return metrics


def compute_per_subject_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_ids: np.ndarray,
    y_proba: np.ndarray | None = None,
    aggregation: str = 'majority',
) -> Dict[str, float]:
    """
    Compute metrics after aggregating predictions per subject.
    
    For each subject, aggregate window-level predictions to a single
    subject-level prediction, then compute metrics.
    
    Parameters
    ----------
    y_true : ndarray
        True labels for each window
    y_pred : ndarray
        Predicted labels for each window
    subject_ids : ndarray
        Subject ID for each window
    y_proba : ndarray, optional
        Predicted probabilities for each window
    aggregation : str
        How to aggregate window predictions per subject:
        - 'majority': Majority vote
        - 'mean_proba': Mean probability (requires y_proba)
        - 'max_proba': Maximum probability (requires y_proba)
        
    Returns
    -------
    metrics : dict
        Subject-level metrics (same structure as compute_comprehensive_metrics)
    """
    unique_subjects = np.unique(subject_ids)
    
    subject_y_true = []
    subject_y_pred = []
    subject_y_proba = [] if y_proba is not None else None
    
    for subj in unique_subjects:
        mask = subject_ids == subj
        subj_true = y_true[mask]
        subj_pred = y_pred[mask]
        
        # Subject-level ground truth (should be consistent)
        # Take mode in case of any inconsistency
        subject_y_true.append(np.bincount(subj_true).argmax())
        
        # Aggregate predictions
        if aggregation == 'majority':
            subject_y_pred.append(np.bincount(subj_pred).argmax())
        elif aggregation == 'mean_proba' and y_proba is not None:
            subj_proba = y_proba[mask]
            if subj_proba.ndim == 2:
                mean_proba = subj_proba[:, 1].mean()
            else:
                mean_proba = subj_proba.mean()
            subject_y_pred.append(int(mean_proba >= 0.5))
            if subject_y_proba is not None:
                subject_y_proba.append(mean_proba)
        elif aggregation == 'max_proba' and y_proba is not None:
            subj_proba = y_proba[mask]
            if subj_proba.ndim == 2:
                max_proba = subj_proba[:, 1].max()
            else:
                max_proba = subj_proba.max()
            subject_y_pred.append(int(max_proba >= 0.5))
            if subject_y_proba is not None:
                subject_y_proba.append(max_proba)
        else:
            # Default to majority vote
            subject_y_pred.append(np.bincount(subj_pred).argmax())
            
    subject_y_true = np.array(subject_y_true)
    subject_y_pred = np.array(subject_y_pred)
    
    if subject_y_proba:
        subject_y_proba = np.array(subject_y_proba)
    else:
        subject_y_proba = None
        
    # Compute metrics at subject level
    metrics = compute_comprehensive_metrics(
        subject_y_true,
        subject_y_pred,
        subject_y_proba
    )
    
    # Add subject count
    metrics['n_subjects'] = len(unique_subjects)
    
    return metrics


def print_evaluation_report(
    metrics: Dict[str, float],
    title: str = "Evaluation Metrics",
) -> None:
    """
    Print formatted evaluation report.
    
    Parameters
    ----------
    metrics : dict
        Metrics dictionary from compute_comprehensive_metrics
    title : str
        Report title
    """
    print("\n" + "=" * 80)
    print(title.upper())
    print("=" * 80)
    
    # Overall metrics
    print(f"Overall Performance:")
    print(f"  Accuracy:     {metrics.get('accuracy', 0):.4f}")
    if 'roc_auc' in metrics and not np.isnan(metrics['roc_auc']):
        print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
    if 'pr_auc' in metrics and not np.isnan(metrics['pr_auc']):
        print(f"  PR-AUC:       {metrics['pr_auc']:.4f}")
    
    # Clinical metrics
    print(f"\nClinical Metrics:")
    print(f"  Sensitivity (Seizure Recall):  {metrics.get('sensitivity', 0):.4f}")
    print(f"  Specificity (Non-seiz Recall): {metrics.get('specificity', 0):.4f}")
    
    # Non-seizure class (class 0)
    print(f"\nNon-Seizure Class (0):")
    print(f"  Precision:    {metrics.get('precision_nonseizure', 0):.4f}")
    print(f"  Recall:       {metrics.get('recall_nonseizure', 0):.4f}")
    print(f"  F1-Score:     {metrics.get('f1_nonseizure', 0):.4f}")
    
    # Seizure class (class 1)
    print(f"\nSeizure Class (1):")
    print(f"  Precision:    {metrics.get('precision_seizure', 0):.4f}")
    print(f"  Recall:       {metrics.get('recall_seizure', 0):.4f}")
    print(f"  F1-Score:     {metrics.get('f1_seizure', 0):.4f}")
    
    # Subject-level metrics if available
    if 'n_subjects' in metrics:
        print(f"\nSubject-Level Aggregation:")
        print(f"  Number of subjects: {metrics['n_subjects']}")
    
    print("=" * 80)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str | None = None,
) -> None:
    """
    Plot ROC curve.
    
    Parameters
    ----------
    y_true : ndarray
        True binary labels
    y_proba : ndarray
        Predicted probabilities for positive class
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    
    # Handle both (n, 2) and (n,) shapes
    if y_proba.ndim == 2:
        y_proba_pos = y_proba[:, 1]
    else:
        y_proba_pos = y_proba
        
    fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
    auc = roc_auc_score(y_true, y_proba_pos)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve: Seizure Classification', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()
