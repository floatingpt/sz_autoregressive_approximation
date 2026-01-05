"""
Post-processing utilities for seizure predictions.
"""
import numpy as np


def temporal_smoothing(
    y_pred: np.ndarray,
    window_size: int = 5,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Apply temporal smoothing via sliding window majority vote.
    
    Reduces false positives by requiring consecutive seizure predictions.
    
    Parameters
    ----------
    y_pred : ndarray
        Binary predictions (0=non-seizure, 1=seizure)
    window_size : int
        Number of consecutive windows to consider (default: 5 = 5 seconds with 1s stride)
    threshold : float
        Fraction of windows that must be seizure to output seizure (default: 0.5)
        
    Returns
    -------
    y_pred_smoothed : ndarray
        Smoothed predictions
    """
    y_smoothed = np.copy(y_pred)
    n = len(y_pred)
    half_window = window_size // 2
    
    for i in range(n):
        # Get window bounds
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        
        # Majority vote in window
        window_votes = y_pred[start:end]
        seizure_fraction = np.mean(window_votes)
        
        y_smoothed[i] = 1 if seizure_fraction >= threshold else 0
    
    return y_smoothed


def per_subject_temporal_smoothing(
    y_pred: np.ndarray,
    subject_ids: np.ndarray,
    window_size: int = 5,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Apply temporal smoothing per-subject (don't smooth across subject boundaries).
    
    Parameters
    ----------
    y_pred : ndarray
        Binary predictions
    subject_ids : ndarray
        Subject ID for each window
    window_size : int
        Smoothing window size
    threshold : float
        Majority vote threshold
        
    Returns
    -------
    y_pred_smoothed : ndarray
        Smoothed predictions
    """
    y_smoothed = np.copy(y_pred)
    
    for subject in np.unique(subject_ids):
        mask = subject_ids == subject
        indices = np.where(mask)[0]
        
        # Sort to ensure temporal order
        indices = indices[np.argsort(indices)]
        
        # Smooth this subject's predictions
        subject_preds = y_pred[indices]
        subject_smoothed = temporal_smoothing(subject_preds, window_size, threshold)
        
        y_smoothed[indices] = subject_smoothed
    
    return y_smoothed
