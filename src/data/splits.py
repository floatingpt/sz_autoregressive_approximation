"""
Subject-level grouped stratified splitting to prevent data leakage.

This module ensures no windows from the same subject appear in both train and test sets.
"""

from __future__ import annotations

from typing import Tuple, Dict
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def create_stratified_subject_split(
    subject_ids: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create train/test split that balances seizure distribution across subjects.
    
    This stratified approach ensures test subjects have similar seizure rates
    to training subjects, reducing class imbalance issues.
    
    Parameters
    ----------
    subject_ids : ndarray, shape (n_samples,)
        Subject identifier for each window
    labels : ndarray, shape (n_samples,)
        Binary labels {0, 1} for each window
    test_size : float
        Proportion of subjects to hold out for testing
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    train_idx : ndarray
        Indices of training samples
    test_idx : ndarray
        Indices of test samples
        
    Notes
    -----
    Algorithm:
    1. Compute seizure rate for each subject
    2. Sort subjects by seizure rate
    3. Alternate assignment to train/test to balance distributions
    4. Shuffle within splits for randomness
    """
    np.random.seed(random_state)
    
    # Get unique subjects and compute their seizure rates
    unique_subjects = np.unique(subject_ids)
    subject_seizure_rates: Dict[str, float] = {}
    subject_indices: Dict[str, np.ndarray] = {}
    
    for subject in unique_subjects:
        subject_mask = subject_ids == subject
        subject_indices[subject] = np.where(subject_mask)[0]
        seizure_rate = labels[subject_mask].mean()
        subject_seizure_rates[subject] = seizure_rate
    
    # Sort subjects by seizure rate
    sorted_subjects = sorted(unique_subjects, key=lambda s: subject_seizure_rates[s])
    
    # Alternate assignment: every other subject goes to test
    # This ensures balanced seizure distribution
    n_test = max(1, int(len(sorted_subjects) * test_size))
    
    # Use alternating pattern with offset to spread across range
    test_subjects = []
    train_subjects = []
    
    # Distribute evenly across sorted list
    step = len(sorted_subjects) / n_test
    for i in range(n_test):
        idx = int(i * step)
        if idx < len(sorted_subjects):
            test_subjects.append(sorted_subjects[idx])
    
    train_subjects = [s for s in sorted_subjects if s not in test_subjects]
    
    # Shuffle for randomness
    np.random.shuffle(train_subjects)
    np.random.shuffle(test_subjects)
    
    # Collect indices
    train_idx = np.concatenate([subject_indices[s] for s in train_subjects])
    test_idx = np.concatenate([subject_indices[s] for s in test_subjects])
    
    # Shuffle indices within splits
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    
    return train_idx, test_idx


def create_subject_grouped_split(
    subject_ids: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create train/test split grouped by subject ID to prevent data leakage.
    
    Parameters
    ----------
    subject_ids : ndarray, shape (n_samples,)
        Subject identifier for each window
    labels : ndarray, shape (n_samples,)
        Binary labels {0, 1} for each window
    test_size : float
        Proportion of subjects to hold out for testing
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    train_idx : ndarray
        Indices of training samples
    test_idx : ndarray
        Indices of test samples
        
    Notes
    -----
    This function ensures that all windows from a given subject belong to
    either the training set OR the test set, never both. This prevents
    temporal/subject-specific information leakage.
    """
    # Use GroupShuffleSplit for subject-level splitting
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    
    # Get single train/test split
    train_idx, test_idx = next(gss.split(
        X=np.zeros(len(subject_ids)),  # Dummy X
        y=labels,
        groups=subject_ids
    ))
    
    return train_idx, test_idx


def verify_no_subject_leakage(
    train_subjects: np.ndarray,
    test_subjects: np.ndarray,
) -> bool:
    """
    Verify that no subject appears in both train and test sets.
    
    Parameters
    ----------
    train_subjects : ndarray
        Subject IDs in training set
    test_subjects : ndarray
        Subject IDs in test set
        
    Returns
    -------
    is_valid : bool
        True if no overlap, False otherwise
    """
    train_set = set(train_subjects)
    test_set = set(test_subjects)
    overlap = train_set & test_set
    
    if overlap:
        print(f"⚠️  WARNING: {len(overlap)} subjects appear in both train and test!")
        print(f"   Overlapping subjects: {sorted(overlap)}")
        return False
    
    return True


def print_split_statistics(
    subject_ids: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """Print detailed statistics about the train/test split."""
    train_subjects = subject_ids[train_idx]
    test_subjects = subject_ids[test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    
    print("\n" + "=" * 80)
    print("SUBJECT-GROUPED SPLIT STATISTICS")
    print("=" * 80)
    
    # Subject counts
    n_unique_train = len(np.unique(train_subjects))
    n_unique_test = len(np.unique(test_subjects))
    print(f"Unique subjects:")
    print(f"  Train: {n_unique_train}")
    print(f"  Test:  {n_unique_test}")
    print(f"  Total: {n_unique_train + n_unique_test}")
    
    # Window counts
    print(f"\nWindow counts:")
    print(f"  Train: {len(train_idx)} windows")
    print(f"  Test:  {len(test_idx)} windows")
    print(f"  Total: {len(train_idx) + len(test_idx)} windows")
    
    # Class balance
    print(f"\nClass balance (Train):")
    train_class_counts = np.bincount(train_labels, minlength=2)
    print(f"  Class 0 (non-seizure): {train_class_counts[0]} ({100*train_class_counts[0]/len(train_labels):.1f}%)")
    print(f"  Class 1 (seizure):     {train_class_counts[1]} ({100*train_class_counts[1]/len(train_labels):.1f}%)")
    
    print(f"\nClass balance (Test):")
    test_class_counts = np.bincount(test_labels, minlength=2)
    print(f"  Class 0 (non-seizure): {test_class_counts[0]} ({100*test_class_counts[0]/len(test_labels):.1f}%)")
    print(f"  Class 1 (seizure):     {test_class_counts[1]} ({100*test_class_counts[1]/len(test_labels):.1f}%)")
    
    # Per-subject seizure rates
    print(f"\nPer-subject seizure rates (Train):")
    unique_train = np.unique(train_subjects)
    train_rates = []
    for subject in sorted(unique_train):
        subject_mask = train_subjects == subject
        subject_labels = train_labels[subject_mask]
        seizure_rate = subject_labels.mean()
        train_rates.append(seizure_rate)
        n_seizures = subject_labels.sum()
        print(f"  {subject}: {n_seizures}/{len(subject_labels)} seizures ({100*seizure_rate:.2f}%)")
    
    print(f"\nPer-subject seizure rates (Test):")
    unique_test = np.unique(test_subjects)
    test_rates = []
    for subject in sorted(unique_test):
        subject_mask = test_subjects == subject
        subject_labels = test_labels[subject_mask]
        seizure_rate = subject_labels.mean()
        test_rates.append(seizure_rate)
        n_seizures = subject_labels.sum()
        print(f"  {subject}: {n_seizures}/{len(subject_labels)} seizures ({100*seizure_rate:.2f}%)")
    
    # Distribution comparison
    if train_rates and test_rates:
        train_imbalance = (1 - train_class_counts[1] / len(train_labels))
        test_imbalance = (1 - test_class_counts[1] / len(test_labels))
        imbalance_ratio = test_imbalance / train_imbalance if train_imbalance > 0 else 1.0
        
        print(f"\nImbalance comparison:")
        print(f"  Train imbalance ratio: 1:{train_class_counts[0]/train_class_counts[1]:.1f}")
        print(f"  Test imbalance ratio:  1:{test_class_counts[0]/test_class_counts[1]:.1f}")
        if imbalance_ratio > 1.5:
            print(f"  ⚠️  Test is {imbalance_ratio:.1f}x more imbalanced than train!")
        else:
            print(f"  ✓ Test and train have similar imbalance ({imbalance_ratio:.2f}x)")
    
    # Verify no leakage
    print(f"\nLeakage check:")
    is_valid = verify_no_subject_leakage(train_subjects, test_subjects)
    if is_valid:
        print("  ✓ No subject leakage detected")
    
    print("=" * 80)
