#!/usr/bin/env python3
"""
Complete Linear Pipeline: Preprocessing → Training → Testing

Single script that efficiently runs the entire MVAR classification workflow:
1. Preprocessing: Load and prepare EEG data with stratified subject splits
2. Training: Extract MVAR features and train classifier
3. Testing: Evaluate and generate comprehensive metrics

Optimized for local execution with memory-efficient processing.

Usage:
    python scripts/run_pipeline.py                    # Full pipeline
    python scripts/run_pipeline.py --skip-preprocessing  # If data already processed
    python scripts/run_pipeline.py --subsample 100    # Quick test with 100 samples/class
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.parameters import DEFAULT_PARAMS
from src.data.paths import PATHS
from src.evaluation.metrics import (
    compute_comprehensive_metrics,
    print_evaluation_report,
)
from src.evaluation.postprocessing import temporal_smoothing
from src.models.mvar_classifier import MVARBinaryClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Standard channel set present in ALL subjects (18 channels)
# These follow the 10-20 International System
STANDARD_CHANNELS = [
    'Fp1', 'Fp2',  # Frontal pole
    'F7', 'F3', 'Fz', 'F4', 'F8',  # Frontal
    'T3', 'C3', 'Cz', 'C4', 'T4',  # Central
    'T5', 'P3', 'Pz', 'P4', 'T6',  # Parietal/Temporal
    'O2',  # Occipital
]


# ============================================================================
# STEP 1: PREPROCESSING
# ============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to file and console."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"pipeline_{timestamp}.log"
    
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def load_and_preprocess_data(
    dataset_root: Path,
    window_sec: float,
    stride_sec: float,
    test_size: float,
    random_state: int,
    output_dir: Path,
    logger: logging.Logger,
    args=None,
) -> Path:
    """
    Load raw EEG data and create train/test splits.
    
    Returns path to manifest.json with data locations.
    """
    try:
        import mne
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "MNE-Python is required for preprocessing EEG data.\n"
            "Install it with: pip install mne\n"
            "Or use the conda environment: conda env create -f environment.yaml"
        )
    import re
    import pandas as pd
    from src.data.splits import create_stratified_subject_split, print_split_statistics
    
    logger.info("=" * 80)
    logger.info("STEP 1: PREPROCESSING")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_root}")
    logger.info(f"Window: {window_sec}s, Stride: {stride_sec}s")
    logger.info(f"Test size: {test_size}, Random seed: {random_state}")
    
    # Suppress MNE output
    mne.set_log_level('WARNING')
    
    # Find all EDF files
    edf_files = sorted(dataset_root.glob("PN*/*.edf"))
    logger.info(f"Found {len(edf_files)} EDF files")
    
    if len(edf_files) == 0:
        raise FileNotFoundError(f"No EDF files found in {dataset_root}")
    
    # Extract windows from all files (memory-efficient: save as we go)
    TIME_RE = re.compile(r"(\d{1,2}\.\d{1,2}\.\d{1,2})")
    TIME_ALT_RE = re.compile(r"(\d{1,2}):(\d{1,2})\.(\d{1,2})")

    # First pass: count windows and track channel counts to pre-allocate arrays
    logger.info("\nPass 1: Counting windows...")
    logger.info(f"Using standardized channel set: {len(STANDARD_CHANNELS)} channels")
    logger.info(f"Channels: {', '.join(STANDARD_CHANNELS)}")
    window_count = 0
    window_metadata = []  # Store (subject, label, file_path, start_sample)
    n_channels = len(STANDARD_CHANNELS)  # Use fixed standard channel count
    sfreq_target = 256
    window_samples = int(window_sec * sfreq_target)
    stride_samples = int(stride_sec * sfreq_target)
    
    for edf_path in tqdm(edf_files, desc="Counting windows"):
        subject_id = edf_path.parent.name
        
        # Load seizure annotations - parse as start/end pairs
        seizure_file = edf_path.parent / f"Seizures-list-{subject_id}.txt"
        if not seizure_file.exists():
            continue
        
        # Parse seizure periods as (start_time, end_time) pairs
        seizure_periods = []
        with open(seizure_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Look for "Seizure start time:" followed by "Seizure end time:"
                if 'Seizure start time:' in line or 'seizure start time:' in line.lower():
                    # Parse start time
                    match = TIME_RE.search(line)
                    if not match:
                        match = TIME_ALT_RE.search(line)
                        if match:
                            h, m, s = match.groups()
                            start_sec = int(h) * 3600 + int(m) * 60 + int(s)
                        else:
                            i += 1
                            continue
                    else:
                        time_str = match.group(1)
                        parts = time_str.split('.')
                        if len(parts) == 3:
                            h, m, s = map(int, parts)
                            start_sec = h * 3600 + m * 60 + s
                        else:
                            i += 1
                            continue
                    
                    # Look for end time in next few lines
                    end_sec = None
                    for j in range(i+1, min(i+5, len(lines))):
                        end_line = lines[j].strip()
                        if 'Seizure end time:' in end_line or 'seizure end time:' in end_line.lower():
                            match = TIME_RE.search(end_line)
                            if not match:
                                match = TIME_ALT_RE.search(end_line)
                                if match:
                                    h, m, s = match.groups()
                                    end_sec = int(h) * 3600 + int(m) * 60 + int(s)
                            else:
                                time_str = match.group(1)
                                parts = time_str.split('.')
                                if len(parts) == 3:
                                    h, m, s = map(int, parts)
                                    end_sec = h * 3600 + m * 60 + s
                            break
                    
                    if end_sec is not None and end_sec > start_sec:
                        seizure_periods.append((start_sec, end_sec))
                
                i += 1
        
        if not seizure_periods:
            continue
        
        try:
            # Quick header check without loading data
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            sfreq_orig = raw.info['sfreq']
            duration_sec = raw.n_times / sfreq_orig
            n_samples = int(duration_sec * sfreq_target)
            
            # Verify all standard channels are present
            # Channel names may have 'EEG ' prefix, so we need to check both formats
            available_channels_normalized = []
            for ch in raw.ch_names:
                ch_clean = ch.replace('EEG ', '').replace('eeg ', '').strip().upper()
                available_channels_normalized.append(ch_clean)
            
            missing_channels = [ch for ch in STANDARD_CHANNELS if ch.upper() not in available_channels_normalized]
            if missing_channels:
                logger.warning(f"{edf_path.name}: Missing standard channels: {', '.join(missing_channels)}")
                continue

            for start_sample in range(0, n_samples - window_samples + 1, stride_samples):
                end_sample = start_sample + window_samples
                start_time = start_sample / sfreq_target
                end_time = end_sample / sfreq_target

                # Check if window overlaps with any seizure period
                is_seizure = any(
                    # Window overlaps if: window_start < seizure_end AND window_end > seizure_start
                    start_time < seizure_end and end_time > seizure_start
                    for seizure_start, seizure_end in seizure_periods
                )

                window_metadata.append((subject_id, 1 if is_seizure else 0, str(edf_path), start_sample))
                window_count += 1

        except Exception as e:
            logger.warning(f"Failed to process {edf_path}: {e}")
            continue
    
    logger.info(f"\nCounted {window_count} windows")
    
    if window_count == 0:
        raise ValueError(
            "No windows were extracted from any EDF files. Possible issues:\n"
            "  1. No EDF files found with valid seizure annotations\n"
            "  2. All files are missing one or more standard channels\n"
            "  3. Files may have been filtered out due to errors\n"
            f"  Standard channels required: {', '.join(STANDARD_CHANNELS)}\n"
            "  Check the warnings above for files that were skipped."
        )
    
    seizure_count = sum(1 for _, label, _, _ in window_metadata if label == 1)
    logger.info(f"Seizure windows: {seizure_count} ({100*seizure_count/window_count:.1f}%)")

    # Optional early downsampling to avoid huge arrays
    if args.non_seizure_ratio or args.max_non_seizure or args.max_seizure:
        rng = np.random.default_rng(random_state)
        idx_non = [i for i, (_, label, _, _) in enumerate(window_metadata) if label == 0]
        idx_seiz = [i for i, (_, label, _, _) in enumerate(window_metadata) if label == 1]

        # If ratio is specified, it overrides max_non_seizure
        if args.non_seizure_ratio:
            target_non_seizure = int(len(idx_seiz) * args.non_seizure_ratio)
            if len(idx_non) > target_non_seizure:
                idx_non = rng.choice(idx_non, size=target_non_seizure, replace=False).tolist()
                logger.info(f"Using non-seizure ratio {args.non_seizure_ratio}:1 -> {target_non_seizure} non-seizure windows")
        else:
            if args.max_non_seizure and len(idx_non) > args.max_non_seizure:
                idx_non = rng.choice(idx_non, size=args.max_non_seizure, replace=False).tolist()
        
        if args.max_seizure and len(idx_seiz) > args.max_seizure:
            idx_seiz = rng.choice(idx_seiz, size=args.max_seizure, replace=False).tolist()

        keep_idx = sorted(idx_non + idx_seiz)
        window_metadata = [window_metadata[i] for i in keep_idx]
        window_count = len(window_metadata)
        seizure_count = sum(1 for _, label, _, _ in window_metadata if label == 1)
        logger.info(f"After downsampling: {window_count} windows (seizure={seizure_count}, non={window_count - seizure_count}, ratio={((window_count - seizure_count) / seizure_count if seizure_count > 0 else 0):.1f}:1)")
    
    # Pre-allocate memory-mapped arrays
    logger.info("\nPass 2: Extracting window data...")
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    
    # Create memmap arrays
    X_temp = np.memmap(
        temp_dir / 'X_temp.npy',
        dtype=np.float32,
        mode='w+',
        shape=(window_count, n_channels, window_samples)
    )
    y = np.array([label for _, label, _, _ in window_metadata], dtype=np.int8)
    subjects = np.array([subj for subj, _, _, _ in window_metadata], dtype=object)
    
    # Second pass: extract actual windows
    window_idx = 0
    current_file = None
    current_data = None
    
    for subject_id, label, file_path, start_sample in tqdm(window_metadata, desc="Extracting windows"):
        # Load file data only when it changes
        if current_file != file_path:
            if current_data is not None:
                del current_data  # Free memory
            
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            
            # Select and reorder channels to match standard set
            # Handle channel names with 'EEG ' prefix
            # Create mapping from normalized name to actual name in file
            ch_mapping = {}
            for ch in raw.ch_names:
                ch_clean = ch.replace('EEG ', '').replace('eeg ', '').strip()
                ch_mapping[ch_clean.upper()] = ch
            
            # Find channels to pick in order
            channels_to_pick = []
            for std_ch in STANDARD_CHANNELS:
                if std_ch.upper() in ch_mapping:
                    channels_to_pick.append(ch_mapping[std_ch.upper()])
            
            if len(channels_to_pick) != len(STANDARD_CHANNELS):
                logger.warning(f"{Path(file_path).name}: Could not find all standard channels")
                continue
            
            raw.pick_channels(channels_to_pick, ordered=True)
            raw.resample(sfreq_target, verbose=False)
            data = raw.get_data()
            
            # Ensure we have exactly n_channels
            if data.shape[0] != n_channels:
                logger.warning(f"{Path(file_path).name}: Expected {n_channels} channels, got {data.shape[0]}")
                continue
            
            current_data = data
            current_file = file_path
        
        # Extract this specific window
        n_samples = current_data.shape[1]
        end_sample = start_sample + window_samples

        if end_sample <= n_samples:
            X_temp[window_idx] = current_data[:, start_sample:end_sample]
        
        window_idx += 1
    
    # Clean up
    if current_data is not None:
        del current_data
    
    # Use X_temp as X
    X = X_temp
    X.flush()
    
    # Create stratified subject split
    train_idx, test_idx = create_stratified_subject_split(
        subject_ids=subjects,
        labels=y,
        test_size=test_size,
        random_state=random_state,
    )

    # Ensure the test split contains seizure windows at the SUBJECT level
    def _test_has_seizure() -> bool:
        return np.any(y[test_idx] == 1)

    # Retry with different seeds
    max_retries = 10
    retry_seed = random_state
    while not _test_has_seizure() and max_retries > 0:
        retry_seed += 1
        train_idx, test_idx = create_stratified_subject_split(
            subject_ids=subjects,
            labels=y,
            test_size=test_size,
            random_state=retry_seed,
        )
        max_retries -= 1

    # If still no seizures, move a whole seizure subject from train to test (optionally swap a non-seizure subject back)
    if not _test_has_seizure():
        # Map subject -> has_seizure
        subj_has_seizure = {}
        for subj in np.unique(subjects):
            subj_mask = subjects == subj
            subj_has_seizure[subj] = np.any(y[subj_mask] == 1)

        train_subj_set = set(subjects[train_idx])
        test_subj_set = set(subjects[test_idx])

        seizure_train_subj = [s for s in train_subj_set if subj_has_seizure.get(s, False)]
        nonseiz_test_subj = [s for s in test_subj_set if not subj_has_seizure.get(s, False)]

        if seizure_train_subj:
            move_seiz_subj = seizure_train_subj[0]
            move_to_test = np.where(subjects == move_seiz_subj)[0]

            # Optional swap to keep sizes similar
            if nonseiz_test_subj:
                move_back_subj = nonseiz_test_subj[0]
                move_to_train = np.where(subjects == move_back_subj)[0]
            else:
                move_to_train = np.array([], dtype=int)

            train_idx = np.setdiff1d(train_idx, move_to_test, assume_unique=False)
            test_idx = np.union1d(test_idx, move_to_test)

            if move_to_train.size > 0:
                test_idx = np.setdiff1d(test_idx, move_to_train, assume_unique=False)
                train_idx = np.union1d(train_idx, move_to_train)

            logger.warning(f"Test split had no seizures; moved subject {move_seiz_subj} into test (subject-level swap).")
        else:
            logger.warning("No seizure subjects available to move into test; evaluation may be biased.")
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    subject_train = subjects[train_idx]
    
    X_test = X[test_idx]
    y_test = y[test_idx]
    subject_test = subjects[test_idx]
    
    # Print split statistics
    print_split_statistics(
        train_idx=train_idx,
        test_idx=test_idx,
        subject_ids=subjects,
        labels=y,
    )
    
    # Save to disk with compression for space efficiency
    output_dir.mkdir(parents=True, exist_ok=True)
    
    X_train_path = output_dir / "X_train.npy"
    y_train_path = output_dir / "y_train.npy"
    X_test_path = output_dir / "X_test.npy"
    y_test_path = output_dir / "y_test.npy"
    
    logger.info("\nSaving split data to disk...")
    # Create new memmaps and copy data in chunks to avoid RAM spike
    X_train_mm = np.memmap(X_train_path, dtype=np.float32, mode='w+', shape=X_train.shape)
    chunk_size = 1000
    for i in range(0, len(X_train), chunk_size):
        end = min(i + chunk_size, len(X_train))
        X_train_mm[i:end] = X_train[i:end]
    X_train_mm.flush()
    del X_train_mm
    
    X_test_mm = np.memmap(X_test_path, dtype=np.float32, mode='w+', shape=X_test.shape)
    for i in range(0, len(X_test), chunk_size):
        end = min(i + chunk_size, len(X_test))
        X_test_mm[i:end] = X_test[i:end]
    X_test_mm.flush()
    del X_test_mm
    
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test)
    
    # Clean up temp files
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    logger.info(f"Train set: {len(y_train)} samples ({X_train.nbytes / 1024**2:.1f} MB)")
    logger.info(f"Test set: {len(y_test)} samples ({X_test.nbytes / 1024**2:.1f} MB)")
    
    # Create subject window maps
    subject_window_map_train = {}
    for subject in np.unique(subject_train):
        subject_window_map_train[subject] = np.where(subject_train == subject)[0].tolist()
    
    subject_window_map_test = {}
    for subject in np.unique(subject_test):
        subject_window_map_test[subject] = np.where(subject_test == subject)[0].tolist()
    
    # Save manifest
    manifest = {
        "created": datetime.now().isoformat(),
        "sample_rate": 256,
        "window_sec": float(window_sec),
        "stride_sec": float(stride_sec),
        "n_channels": int(X_train.shape[1]),
        "channel_names": STANDARD_CHANNELS,
        "channel_system": "10-20 International System",
        "n_samples_per_window": int(X_train.shape[2]),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_class_0": int(np.sum(y_train == 0)),
        "train_class_1": int(np.sum(y_train == 1)),
        "test_class_0": int(np.sum(y_test == 0)),
        "test_class_1": int(np.sum(y_test == 1)),
        "n_subjects_train": int(len(np.unique(subject_train))),
        "n_subjects_test": int(len(np.unique(subject_test))),
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
        "X_train": str(X_train_path),
        "y_train": str(y_train_path),
        "X_test": str(X_test_path),
        "y_test": str(y_test_path),
        "subject_window_map_train": subject_window_map_train,
        "subject_window_map_test": subject_window_map_test,
        "split_method": "stratified_by_subject",
        "test_size": float(test_size),
        "random_seed": int(random_state),
    }
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"\n✓ Preprocessing complete. Manifest saved to {manifest_path}")
    
    return manifest_path


# ============================================================================
# STEP 2: TRAINING
# ============================================================================

def train_mvar_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    mvar_order: int,
    n_basis: int,
    basis_type: str,
    regularization: float,
    n_time_points: int,
    threshold_metric: str,
    seizure_weight: float | None,
    logger: logging.Logger,
) -> MVARBinaryClassifier:
    """
    Train MVAR classifier on training data.
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: TRAINING MVAR CLASSIFIER")
    logger.info("=" * 80)
    logger.info(f"MVAR order: {mvar_order}")
    logger.info(f"Basis functions: {n_basis} ({basis_type})")
    logger.info(f"Regularization: {regularization}")
    logger.info(f"Time points for supremum: {n_time_points}")
    logger.info(f"Training samples: {len(y_train)} ({np.sum(y_train==0)} non-seizure, {np.sum(y_train==1)} seizure)")
    
    classifier = MVARBinaryClassifier(
        mvar_order=mvar_order,
        n_basis=n_basis,
        basis_type=basis_type,
        regularization=regularization,
        n_time_points=n_time_points,
        n_grid_points=100,
        threshold_metric=threshold_metric,
        seizure_weight=seizure_weight,
    )
    
    classifier.fit(X_train, y_train)
    
    logger.info("\n✓ Training complete")
    
    return classifier


# ============================================================================
# STEP 3: TESTING
# ============================================================================

def evaluate_classifier(
    classifier: MVARBinaryClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    logger: logging.Logger,
    fold_num: int | None = None,
    temporal_smoothing_window: int = 0,
) -> Dict:
    """
    Evaluate classifier on test data and save results.
    """
    fold_str = f" (Fold {fold_num})" if fold_num is not None else ""
    logger.info("\n" + "=" * 80)
    logger.info(f"EVALUATION{fold_str}")
    logger.info("=" * 80)
    logger.info(f"Test samples: {len(y_test)} ({np.sum(y_test==0)} non-seizure, {np.sum(y_test==1)} seizure)")
    
    # Predictions
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)
    
    # Apply temporal smoothing if requested
    if temporal_smoothing_window > 0:
        logger.info(f"\nApplying temporal smoothing (window={temporal_smoothing_window})...")
        y_pred_raw = y_pred.copy()
        y_pred = temporal_smoothing(y_pred, window_size=temporal_smoothing_window, threshold=0.5)
        
        # Log impact
        n_changed = np.sum(y_pred_raw != y_pred)
        logger.info(f"  Changed {n_changed}/{len(y_pred)} predictions ({100*n_changed/len(y_pred):.1f}%)")
    
    # Window-level metrics
    logger.info("\nWindow-Level Performance:")
    window_metrics = compute_comprehensive_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
    )
    print_evaluation_report(window_metrics, title=f"Window-Level Performance{fold_str}")
    
    # Generate visualizations (only for final evaluation, not per-fold)
    if fold_num is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = output_dir.parent / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('ROC Curve - Seizure Detection', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        roc_path = figures_dir / 'roc_curve.png'
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✓ ROC curve saved: {roc_path}")
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        baseline = np.sum(y_test == 1) / len(y_test)
        plt.axhline(y=baseline, color='navy', lw=2, linestyle='--', label=f'Baseline ({baseline:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Sensitivity)', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve - Seizure Detection', fontsize=14, fontweight='bold')
        plt.legend(loc="upper right", fontsize=11)
        plt.grid(alpha=0.3)
        pr_path = figures_dir / 'precision_recall_curve.png'
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✓ PR curve saved: {pr_path}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Seizure', 'Seizure'])
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        ax.set_title('Confusion Matrix - Window-Level Predictions', fontsize=14, fontweight='bold')
        cm_path = figures_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✓ Confusion matrix saved: {cm_path}")
    
    # Save results
    if fold_num is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'window_level': {k: float(v) if not np.isnan(v) else None for k, v in window_metrics.items()},
            'classifier_config': {
                'mvar_order': classifier.feature_extractor.mvar_order,
                'n_basis': classifier.feature_extractor.n_basis,
                'basis_type': classifier.feature_extractor.basis_type,
                'regularization': classifier.feature_extractor.regularization,
                'threshold': float(classifier.threshold_),
                'class_averages': {int(k): float(v) for k, v in classifier.class_averages_.items()},
            }
        }
        
        results_path = output_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✓ Results saved to {output_dir}")
        logger.info(f"  - {results_path}")
    
    return window_metrics


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Pipeline control
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip preprocessing step (use existing data in artifacts/)'
    )
    parser.add_argument(
        '--subsample',
        type=int,
        default=None,
        help='Subsample N samples per class (for quick testing)'
    )
    parser.add_argument(
        '--balance',
        type=str,
        choices=['none', 'undersample', 'oversample'],
        default='none',
        help='Class balancing strategy for train split (default: none)'
    )
    parser.add_argument(
        '--balance-test',
        action='store_true',
        help='Also balance test split (rarely needed; default: False)'
    )
    parser.add_argument(
        '--max-non-seizure',
        type=int,
        default=None,
        help='Cap non-seizure windows before saving to disk (downsample majority early to save RAM)'
    )
    parser.add_argument(
        '--max-seizure',
        type=int,
        default=None,
        help='Cap seizure windows before saving to disk'
    )
    parser.add_argument(
        '--non-seizure-ratio',
        type=float,
        default=None,
        help='Ratio of non-seizure to seizure windows (e.g., 2.0 = 2:1 ratio). Overrides --max-non-seizure if both specified.'
    )
    
    # Preprocessing parameters
    parser.add_argument(
        '--dataset-root',
        type=Path,
        default=PATHS["project_root"] / DEFAULT_PARAMS["dataset_root"],
        help='Path to raw EEG dataset'
    )
    parser.add_argument(
        '--window-sec',
        type=float,
        default=DEFAULT_PARAMS["window_sec"],
        help='Window length in seconds (default: 2.0)'
    )
    parser.add_argument(
        '--stride-sec',
        type=float,
        default=DEFAULT_PARAMS["stride_sec"],
        help='Stride length in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=DEFAULT_PARAMS["test_size"],
        help='Test set fraction (default: 0.3)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_PARAMS["random_seed"],
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=1,
        help='Number of cross-validation folds (default: 1 = no CV, single train/test split)'
    )
    
    # MVAR classifier parameters
    parser.add_argument(
        '--mvar-order',
        type=int,
        default=DEFAULT_PARAMS.get('mvar_order', 3),
        help='MVAR model order (default: 3)'
    )
    parser.add_argument(
        '--n-basis',
        type=int,
        default=DEFAULT_PARAMS.get('n_basis', 10),
        help='Number of basis functions (default: 10)'
    )
    parser.add_argument(
        '--basis-type',
        type=str,
        choices=['bspline', 'polynomial'],
        default=DEFAULT_PARAMS.get('basis_type', 'bspline'),
        help='Basis function type (default: bspline)'
    )
    parser.add_argument(
        '--regularization',
        type=float,
        default=DEFAULT_PARAMS.get('regularization', 0.1),
        help='Ridge regularization strength (default: 0.1)'
    )
    parser.add_argument(
        '--n-time-points',
        type=int,
        default=DEFAULT_PARAMS.get('n_time_points', 50),
        help='Time resolution for computing supremum (default: 50)'
    )
    parser.add_argument(
        '--threshold-metric',
        type=str,
        choices=['f1_seizure', 'youden'],
        default='f1_seizure',
        help='Metric to optimize threshold: seizure F1 or Youden J (weighted)'
    )
    parser.add_argument(
        '--seizure-weight',
        type=float,
        default=None,
        help='Optional weight (0-1) emphasizing seizure recall when using Youden or F1 scoring'
    )
    parser.add_argument(
        '--temporal-smoothing',
        type=int,
        default=0,
        help='Apply temporal smoothing with window size N (0=disabled, 5=recommended for 1s stride). Reduces isolated false positives.'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PATHS['features'],
        help='Output directory for artifacts and results'
    )
    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Save trained classifier to disk'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    logger.info("=" * 80)
    logger.info("MVAR CLASSIFICATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Step 1: Preprocessing
    if args.skip_preprocessing:
        logger.info("\nSkipping preprocessing (using existing data)")
        manifest_path = args.output_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. "
                "Run without --skip-preprocessing first."
            )
    else:
        manifest_path = load_and_preprocess_data(
            dataset_root=args.dataset_root,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            test_size=args.test_size,
            random_state=args.seed,
            output_dir=args.output_dir,
            logger=logger,
            args=args,
        )
    
    # Load preprocessed data
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Load with memmap (X arrays are stored as memmap files, not .npy format)
    X_train = np.memmap(
        args.output_dir / "X_train.npy", 
        dtype=np.float32, 
        mode='r',
        shape=tuple(manifest['X_train_shape'])
    )
    y_train = np.load(args.output_dir / "y_train.npy")
    X_test = np.memmap(
        args.output_dir / "X_test.npy", 
        dtype=np.float32, 
        mode='r',
        shape=tuple(manifest['X_test_shape'])
    )
    y_test = np.load(args.output_dir / "y_test.npy")
    
    # Reconstruct subject arrays from manifest
    subject_train = np.empty(len(y_train), dtype=object)
    for subject_id, indices in manifest['subject_window_map_train'].items():
        subject_train[np.array(indices)] = subject_id
    
    subject_test = np.empty(len(y_test), dtype=object)
    for subject_id, indices in manifest['subject_window_map_test'].items():
        subject_test[np.array(indices)] = subject_id
    
    # Optional subsampling for quick testing
    if args.subsample:
        logger.info(f"\n⚡ Subsampling {args.subsample} samples per class for quick testing")
        from sklearn.model_selection import StratifiedShuffleSplit
        
        sss = StratifiedShuffleSplit(n_splits=1, train_size=2*args.subsample, random_state=42)
        train_idx, _ = next(sss.split(np.zeros(len(y_train)), y_train))
        test_idx, _ = next(sss.split(np.zeros(len(y_test)), y_test))
        
        # Force copy to RAM for subsampled data (small enough to fit)
        X_train = np.array(X_train[train_idx])
        y_train = y_train[train_idx]
        subject_train = subject_train[train_idx]
        
        X_test = np.array(X_test[test_idx])
        y_test = y_test[test_idx]
        subject_test = subject_test[test_idx]
        
        logger.info(f"  Train: {len(y_train)} samples ({np.bincount(y_train)})")
        logger.info(f"  Test:  {len(y_test)} samples ({np.bincount(y_test)})")
    
    # Optional class balancing
    def _balance_split(X, y, subjects, strategy, rng, label="train"):
        if strategy == 'none':
            return X, y, subjects
        uniq, counts = np.unique(y, return_counts=True)
        if len(uniq) < 2:
            logger.warning(f"Cannot balance {label} split: only one class present")
            return X, y, subjects
        minority = uniq[np.argmin(counts)]
        majority = uniq[np.argmax(counts)]
        n_min = counts[np.argmin(counts)]
        n_maj = counts[np.argmax(counts)]
        if strategy == 'undersample':
            keep_min_idx = np.where(y == minority)[0]
            keep_maj_idx = rng.choice(np.where(y == majority)[0], size=n_min, replace=False)
            keep_idx = np.concatenate([keep_min_idx, keep_maj_idx])
        else:  # oversample
            keep_min_idx = np.where(y == minority)[0]
            keep_maj_idx = np.where(y == majority)[0]
            extra_min = rng.choice(keep_min_idx, size=n_maj - n_min, replace=True) if n_maj > n_min else np.array([], dtype=int)
            keep_idx = np.concatenate([keep_maj_idx, keep_min_idx, extra_min]) if n_maj > n_min else np.concatenate([keep_maj_idx, keep_min_idx])
        rng.shuffle(keep_idx)
        Xb = X[keep_idx]
        yb = y[keep_idx]
        sb = subjects[keep_idx]
        logger.info(f"Balanced {label}: strategy={strategy}, counts before={dict(zip(uniq, counts))}, after={dict(zip(*np.unique(yb, return_counts=True)))}")
        return Xb, yb, sb
    
    rng = np.random.default_rng(args.seed)
    X_train, y_train, subject_train = _balance_split(X_train, y_train, subject_train, args.balance, rng, label="train")
    if args.balance_test:
        X_test, y_test, subject_test = _balance_split(X_test, y_test, subject_test, args.balance, rng, label="test")
    
    # Step 2 & 3: Cross-Validation or Single Split
    if args.cv_folds > 1:
        logger.info("\n" + "=" * 80)
        logger.info(f"CROSS-VALIDATION: {args.cv_folds} Folds (Subject-Stratified)")
        logger.info("=" * 80)
        
        # Combine train and test for CV
        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
        subjects_all = np.concatenate([subject_train, subject_test], axis=0)
        
        # Group by subject for stratified subject-level CV
        unique_subjects = np.unique(subjects_all)
        subject_labels = np.array([np.bincount(y_all[subjects_all == s]).argmax() for s in unique_subjects])
        
        # Create subject-level folds
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        
        fold_metrics = []
        for fold_num, (train_subj_idx, test_subj_idx) in enumerate(skf.split(unique_subjects, subject_labels), 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Fold {fold_num}/{args.cv_folds}")
            logger.info(f"{'='*60}")
            
            # Get subject IDs for this fold
            train_subjects_fold = unique_subjects[train_subj_idx]
            test_subjects_fold = unique_subjects[test_subj_idx]
            
            # Get window indices for these subjects
            train_mask = np.isin(subjects_all, train_subjects_fold)
            test_mask = np.isin(subjects_all, test_subjects_fold)
            
            X_train_fold = X_all[train_mask]
            y_train_fold = y_all[train_mask]
            X_test_fold = X_all[test_mask]
            y_test_fold = y_all[test_mask]
            
            logger.info(f"Train: {len(y_train_fold)} windows from {len(train_subjects_fold)} subjects ")
            logger.info(f"       ({np.sum(y_train_fold==0)} non-seizure, {np.sum(y_train_fold==1)} seizure)")
            logger.info(f"Test:  {len(y_test_fold)} windows from {len(test_subjects_fold)} subjects ")
            logger.info(f"       ({np.sum(y_test_fold==0)} non-seizure, {np.sum(y_test_fold==1)} seizure)")
            
            # Train classifier for this fold
            classifier_fold = train_mvar_classifier(
                X_train=X_train_fold,
                y_train=y_train_fold,
                mvar_order=args.mvar_order,
                n_basis=args.n_basis,
                basis_type=args.basis_type,
                regularization=args.regularization,
                n_time_points=args.n_time_points,
                threshold_metric=args.threshold_metric,
                seizure_weight=args.seizure_weight,
                logger=logger,
            )
            
            # Evaluate on test fold
            metrics_fold = evaluate_classifier(
                classifier=classifier_fold,
                X_test=X_test_fold,
                y_test=y_test_fold,
                output_dir=PATHS['results'],
                logger=logger,
                fold_num=fold_num,
                temporal_smoothing_window=args.temporal_smoothing,
            )
            fold_metrics.append(metrics_fold)
        
        # Aggregate CV results
        logger.info("\n" + "=" * 80)
        logger.info("CROSS-VALIDATION SUMMARY")
        logger.info("=" * 80)
        
        metric_names = list(fold_metrics[0].keys())
        cv_summary = {}
        for metric in metric_names:
            values = [fold[metric] for fold in fold_metrics if not np.isnan(fold[metric])]
            if values:
                cv_summary[f"{metric}_mean"] = float(np.mean(values))
                cv_summary[f"{metric}_std"] = float(np.std(values))
        
        logger.info("\nAverage Performance Across Folds:")
        for metric in metric_names:
            if f"{metric}_mean" in cv_summary:
                mean = cv_summary[f"{metric}_mean"]
                std = cv_summary[f"{metric}_std"]
                logger.info(f"  {metric:25s}: {mean:.4f} ± {std:.4f}")
        
        # Save CV summary
        cv_results_path = PATHS['results'] / 'cv_results.json'
        with open(cv_results_path, 'w') as f:
            json.dump({
                'cv_folds': args.cv_folds,
                'fold_metrics': fold_metrics,
                'summary': cv_summary,
            }, f, indent=2)
        logger.info(f"\n✓ CV results saved: {cv_results_path}")
        
        # Train final model on all data for visualization
        logger.info("\n" + "=" * 80)
        logger.info("FINAL MODEL: Training on Full Dataset")
        logger.info("=" * 80)
        classifier = train_mvar_classifier(
            X_train=X_all,
            y_train=y_all,
            mvar_order=args.mvar_order,
            n_basis=args.n_basis,
            basis_type=args.basis_type,
            regularization=args.regularization,
            n_time_points=args.n_time_points,
            threshold_metric=args.threshold_metric,
            seizure_weight=args.seizure_weight,
            logger=logger,
        )
        # Generate visualizations on full dataset
        _ = evaluate_classifier(
            classifier=classifier,
            X_test=X_all,
            y_test=y_all,
            output_dir=PATHS['results'],
            logger=logger,
            fold_num=None,
            temporal_smoothing_window=args.temporal_smoothing,
        )
    else:
        # Single train/test split
        logger.info("\n" + "=" * 80)
        logger.info("SINGLE TRAIN/TEST SPLIT")
        logger.info("=" * 80)
        
        classifier = train_mvar_classifier(
            X_train=X_train,
            y_train=y_train,
            mvar_order=args.mvar_order,
            n_basis=args.n_basis,
            basis_type=args.basis_type,
            regularization=args.regularization,
            n_time_points=args.n_time_points,
            threshold_metric=args.threshold_metric,
            seizure_weight=args.seizure_weight,
            logger=logger,
        )
        
        results = evaluate_classifier(
            classifier=classifier,
            X_test=X_test,
            y_test=y_test,
            output_dir=PATHS['results'],
            logger=logger,
            fold_num=None,
            temporal_smoothing_window=args.temporal_smoothing,
        )
    
    # Save trained model
    if args.save_model:
        model_path = PATHS['results'] / 'mvar_classifier.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'classifier': classifier,
                'config': vars(args),
                'manifest': manifest,
            }, f)
        logger.info(f"\n✓ Model saved to {model_path}")
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
