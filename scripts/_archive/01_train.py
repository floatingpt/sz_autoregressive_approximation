"""
Enhanced MVAR-based Binary Classification with Logistic Regression.

Replaces median-based classifier with regularized logistic regression.
Uses multi-dimensional MVAR features for improved discrimination.
Includes comprehensive evaluation with clinically relevant metrics.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.parameters import DEFAULT_PARAMS
from src.data.paths import PATHS
from src.models.enhanced_features import EnhancedMVARFeatureExtractor
from src.evaluation.metrics import (
    compute_comprehensive_metrics,
    compute_per_subject_metrics,
    print_evaluation_report,
    plot_roc_curve,
)


def load_data():
    """Load preprocessed data with subject IDs."""
    manifest_path = PATHS["manifest"]
    
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            "Run scripts/00_preprocessing.py first."
        )
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print("=" * 80)
    print("LOADING PREPROCESSED DATA")
    print("=" * 80)
    print(f"Split method: {manifest.get('split_method', 'unknown')}")
    print(f"Sample rate: {manifest['sample_rate']} Hz")
    print(f"Window length: {manifest['window_sec']} s")
    print(f"Channels: {manifest['n_channels']}")
    print(f"Train samples: {manifest['n_train']} ({manifest['train_class_0']} non-seizure, {manifest['train_class_1']} seizure)")
    print(f"Test samples:  {manifest['n_test']} ({manifest['test_class_0']} non-seizure, {manifest['test_class_1']} seizure)")
    print(f"Train subjects: {manifest['n_subjects_train']}")
    print(f"Test subjects:  {manifest['n_subjects_test']}")
    print()
    
    # Load arrays
    X_train_shape = tuple(manifest['X_train_shape'])
    X_test_shape = tuple(manifest['X_test_shape'])
    
    X_train = np.memmap(PATHS["X_train"], dtype=np.float32, mode='r', shape=X_train_shape)
    y_train = np.memmap(PATHS["y_train"], dtype=np.int8, mode='r', shape=(manifest['n_train'],))
    X_test = np.memmap(PATHS["X_test"], dtype=np.float32, mode='r', shape=X_test_shape)
    y_test = np.memmap(PATHS["y_test"], dtype=np.int8, mode='r', shape=(manifest['n_test'],))
    
    # Reconstruct subject IDs from window map (more efficient than reading files)
    subject_train = np.empty(manifest['n_train'], dtype=object)
    for subject, indices in manifest['subject_window_map_train'].items():
        subject_train[indices] = subject
    
    subject_test = np.empty(manifest['n_test'], dtype=object)
    for subject, indices in manifest['subject_window_map_test'].items():
        subject_test[indices] = subject
    
    return X_train, y_train, subject_train, X_test, y_test, subject_test, manifest


def train_classifier(
    X_train,
    y_train,
    mvar_order=3,
    n_basis=10,
    basis_type='bspline',
    regularization=0.1,
    upper_lag_range=None,
    n_time_points=50,
    include_per_lag=False,
    C=1.0,
    max_iter=1000,
):
    """Train enhanced MVAR classifier with logistic regression."""
    print("=" * 80)
    print("TRAINING ENHANCED MVAR CLASSIFIER")
    print("=" * 80)
    print(f"MVAR Configuration:")
    print(f"  Order:         {mvar_order}")
    print(f"  Basis:         {n_basis} ({basis_type})")
    print(f"  Regularization: {regularization}")
    print(f"  Upper lags:    {upper_lag_range}")
    print(f"  Time points:   {n_time_points}")
    print(f"  Per-lag feats: {include_per_lag}")
    print(f"\nClassifier Configuration:")
    print(f"  Type:          Logistic Regression")
    print(f"  C (inverse reg): {C}")
    print(f"  Class weight:  balanced")
    print()
    
    # Step 1: Extract MVAR features
    print("Step 1: Extracting MVAR features...")
    feature_extractor = EnhancedMVARFeatureExtractor(
        mvar_order=mvar_order,
        n_basis=n_basis,
        basis_type=basis_type,
        regularization=regularization,
        upper_lag_range=upper_lag_range,
        n_time_points=n_time_points,
        include_per_lag=include_per_lag,
    )
    
    X_train_features = feature_extractor.transform(X_train, verbose=True)
    feature_names = feature_extractor.get_feature_names()
    
    print(f"\n✓ Extracted features shape: {X_train_features.shape}")
    print(f"  Feature names: {feature_names}")
    
    # Step 2: Train classifier pipeline
    print("\nStep 2: Training logistic regression classifier...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            penalty='l2',
            C=C,
            class_weight='balanced',
            max_iter=max_iter,
            random_state=42,
            solver='lbfgs',
        ))
    ])
    
    pipeline.fit(X_train_features, y_train)
    
    # Training accuracy
    train_pred = pipeline.predict(X_train_features)
    train_acc = (train_pred == y_train).mean()
    
    print(f"\n✓ Training completed")
    print(f"  Training accuracy: {train_acc:.4f}")
    
    return feature_extractor, pipeline, feature_names


def evaluate_classifier(
    feature_extractor,
    pipeline,
    X_test,
    y_test,
    subject_test,
    output_dir,
):
    """Evaluate classifier with comprehensive metrics."""
    print("\n" + "=" * 80)
    print("EVALUATING ON TEST SET")
    print("=" * 80)
    
    # Extract features
    print("Extracting features from test set...")
    X_test_features = feature_extractor.transform(X_test, verbose=True)
    
    # Predictions
    print("\nMaking predictions...")
    y_pred = pipeline.predict(X_test_features)
    y_proba = pipeline.predict_proba(X_test_features)
    
    # Window-level metrics
    print("\n" + "-" * 80)
    print("WINDOW-LEVEL METRICS")
    print("-" * 80)
    
    window_metrics = compute_comprehensive_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
    )
    
    print_evaluation_report(window_metrics, title="Window-Level Performance")
    
    # Subject-level metrics
    print("\n" + "-" * 80)
    print("SUBJECT-LEVEL METRICS (MAJORITY VOTE)")
    print("-" * 80)
    
    subject_metrics = compute_per_subject_metrics(
        y_true=y_test,
        y_pred=y_pred,
        subject_ids=subject_test,
        y_proba=y_proba,
        aggregation='majority',
    )
    
    print_evaluation_report(subject_metrics, title="Subject-Level Performance (Majority Vote)")
    
    # Subject-level with mean probability
    subject_metrics_mean = compute_per_subject_metrics(
        y_true=y_test,
        y_pred=y_pred,
        subject_ids=subject_test,
        y_proba=y_proba,
        aggregation='mean_proba',
    )
    
    print_evaluation_report(subject_metrics_mean, title="Subject-Level Performance (Mean Probability)")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'window_level': {k: float(v) if not np.isnan(v) else None for k, v in window_metrics.items()},
        'subject_level_majority': {k: float(v) if not np.isnan(v) else None for k, v in subject_metrics.items()},
        'subject_level_mean_proba': {k: float(v) if not np.isnan(v) else None for k, v in subject_metrics_mean.items()},
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot ROC curve
    plot_roc_curve(
        y_true=y_test,
        y_proba=y_proba,
        save_path=str(output_dir / 'roc_curve.png'),
    )
    
    print(f"\n✓ Results saved to {output_dir}")
    
    return window_metrics, subject_metrics


def main():
    """Main training and evaluation pipeline (enhanced)."""
    parser = argparse.ArgumentParser(description=__doc__)
    
    # MVAR parameters
    parser.add_argument('--mvar-order', type=int, default=DEFAULT_PARAMS.get('mvar_order', 5))
    parser.add_argument('--n-basis', type=int, default=DEFAULT_PARAMS.get('n_basis', 10))
    parser.add_argument('--basis-type', type=str, default=DEFAULT_PARAMS.get('basis_type', 'bspline'))
    parser.add_argument('--regularization', type=float, default=DEFAULT_PARAMS.get('regularization', 0.001))
    parser.add_argument('--n-time-points', type=int, default=DEFAULT_PARAMS.get('n_time_points', 50))
    parser.add_argument('--include-per-lag', action='store_true', help='Include per-lag features')
    
    # Classifier parameters
    parser.add_argument('--C', type=float, default=1.0, help='Inverse regularization strength for LogisticRegression')
    parser.add_argument('--max-iter', type=int, default=1000, help='Maximum iterations for LogisticRegression')
    
    # Data subsampling (optional)
    parser.add_argument('--subsample', type=int, default=None, help='Subsample N samples per class for testing')
    
    # Output
    parser.add_argument('--output-dir', type=Path, default=PATHS['results'], help='Output directory')
    parser.add_argument('--save-model', action='store_true', help='Save trained model to disk')
    
    args = parser.parse_args()
    
    # Load data
    X_train, y_train, subject_train, X_test, y_test, subject_test, manifest = load_data()
    
    # Optional subsampling
    if args.subsample:
        print(f"\n⚡ Subsampling {args.subsample} samples per class for testing...")
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=2*args.subsample, random_state=42)
        train_sub_idx, _ = next(sss.split(np.zeros(len(y_train)), y_train))
        
        X_train = np.array([X_train[i] for i in train_sub_idx])
        y_train = y_train[train_sub_idx]
        subject_train = subject_train[train_sub_idx]
        
        print(f"  ✓ New train size: {len(y_train)}")
        print(f"  Classes: {np.bincount(y_train)}")
        
        # Optionally subsample test set too
        test_sub_idx, _ = next(sss.split(np.zeros(len(y_test)), y_test))
        X_test = np.array([X_test[i] for i in test_sub_idx])
        y_test = y_test[test_sub_idx]
        subject_test = subject_test[test_sub_idx]
        
        print(f"  ✓ New test size: {len(y_test)}")
        print(f"  Classes: {np.bincount(y_test)}")
    
    # Train classifier
    feature_extractor, pipeline, feature_names = train_classifier(
        X_train=X_train,
        y_train=y_train,
        mvar_order=args.mvar_order,
        n_basis=args.n_basis,
        basis_type=args.basis_type,
        regularization=args.regularization,
        upper_lag_range=DEFAULT_PARAMS.get('upper_lag_range'),
        n_time_points=args.n_time_points,
        include_per_lag=args.include_per_lag,
        C=args.C,
        max_iter=args.max_iter,
    )
    
    # Evaluate
    window_metrics, subject_metrics = evaluate_classifier(
        feature_extractor=feature_extractor,
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        subject_test=subject_test,
        output_dir=args.output_dir,
    )
    
    # Save model
    if args.save_model:
        model_path = args.output_dir / 'mvar_classifier.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'feature_extractor': feature_extractor,
                'pipeline': pipeline,
                'feature_names': feature_names,
                'config': vars(args),
            }, f)
        print(f"\n✓ Model saved to {model_path}")
    
    print("\n" + "=" * 80)
    print("✓ TRAINING AND EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

