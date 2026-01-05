"""
Utility functions for loading and working with MVAR artifacts.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.data.paths import PATHS


def load_preprocessed_data(
    train_only: bool = False,
    test_only: bool = False,
    load_into_memory: bool = False,
) -> Tuple:
    """
    Load preprocessed EEG data from memmap files.
    
    Parameters
    ----------
    train_only : bool
        Load only training data
    test_only : bool
        Load only test data
    load_into_memory : bool
        If True, load entire arrays into RAM (default: memory-mapped)
        
    Returns
    -------
    data : tuple
        If train_only: (X_train, y_train, manifest)
        If test_only: (X_test, y_test, manifest)
        Otherwise: (X_train, y_train, X_test, y_test, manifest)
    """
    manifest_path = PATHS["manifest"]
    
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            "Run scripts/00_preprocessing.py first."
        )
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Extract shape information from manifest
    # Support two manifest formats: balanced (with train_per_class/test_per_class)
    # and legacy (with explicit n_train/n_test).
    if (
        'train_per_class' in manifest and manifest.get('train_per_class') is not None
        and 'test_per_class' in manifest and manifest.get('test_per_class') is not None
    ):
        n_train = 2 * manifest['train_per_class']
        n_test = 2 * manifest['test_per_class']
    else:
        n_train = manifest.get('n_train')
        n_test = manifest.get('n_test')
    n_channels = manifest['n_channels']
    win_samples = int(manifest['window_sec'] * manifest['sample_rate'])
    
    mode = 'r' if not load_into_memory else 'c'  # 'c' = copy-on-write
    
    if train_only:
        X_train = np.memmap(PATHS["X_train"], dtype=np.float32, mode=mode,
                           shape=(n_train, n_channels, win_samples))
        y_train = np.memmap(PATHS["y_train"], dtype=np.int8, mode=mode, shape=(n_train,))
        if load_into_memory:
            X_train = np.array(X_train)
            y_train = np.array(y_train)
        return X_train, y_train, manifest
    
    if test_only:
        X_test = np.memmap(PATHS["X_test"], dtype=np.float32, mode=mode,
                          shape=(n_test, n_channels, win_samples))
        y_test = np.memmap(PATHS["y_test"], dtype=np.int8, mode=mode, shape=(n_test,))
        if load_into_memory:
            X_test = np.array(X_test)
            y_test = np.array(y_test)
        return X_test, y_test, manifest
    
    X_train = np.memmap(PATHS["X_train"], dtype=np.float32, mode=mode,
                       shape=(n_train, n_channels, win_samples))
    y_train = np.memmap(PATHS["y_train"], dtype=np.int8, mode=mode, shape=(n_train,))
    X_test = np.memmap(PATHS["X_test"], dtype=np.float32, mode=mode,
                      shape=(n_test, n_channels, win_samples))
    y_test = np.memmap(PATHS["y_test"], dtype=np.int8, mode=mode, shape=(n_test,))
    
    if load_into_memory:
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
    
    return X_train, y_train, X_test, y_test, manifest


def print_data_summary(manifest: Dict) -> None:
    """Print summary statistics from manifest."""
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Sample rate:         {manifest['sample_rate']} Hz")
    print(f"Window duration:     {manifest['window_sec']} s")
    print(f"Number of channels:  {manifest['n_channels']}")
    print(f"Train per class:     {manifest['train_per_class']}")
    print(f"Test per class:      {manifest['test_per_class']}")
    print(f"Total train:         {2 * manifest['train_per_class']}")
    print(f"Total test:          {2 * manifest['test_per_class']}")
    print("=" * 60)
