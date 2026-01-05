"""
Nested Grouped Cross-Validation for MVAR Classifier.

Implements nested CV with subject grouping to:
- Outer loop: Estimate generalization performance
- Inner loop: Hyperparameter tuning

Ensures no subject leakage across folds.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.utils.class_weight import compute_sample_weight

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.parameters import DEFAULT_PARAMS
from src.data.paths import PATHS
from src.models.enhanced_features import EnhancedMVARFeatureExtractor
from src.evaluation.metrics import compute_comprehensive_metrics, print_evaluation_report


def load_data(subsample=None):
    """Load training data with subject IDs.
    
    Parameters
    ----------
    subsample : int, optional
        If provided, subsample data BEFORE loading into memory
    """
    manifest_path = PATHS["manifest"]
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print("=" * 80)
    print("LOADING DATA FOR NESTED CV")
    print("=" * 80)
    print(f"Train samples: {manifest['n_train']}")
    print(f"Train subjects: {manifest['n_subjects_train']}")
    
    # Load memmap first
    X_train_shape = tuple(manifest['X_train_shape'])
    X_train_mmap = np.memmap(PATHS["X_train"], dtype=np.float32, mode='r', shape=X_train_shape)
    y_train_mmap = np.memmap(PATHS["y_train"], dtype=np.int8, mode='r', shape=(manifest['n_train'],))
    
    # Reconstruct subject IDs from window map
    subject_train = np.empty(manifest['n_train'], dtype=object)
    for subject, indices in manifest['subject_window_map_train'].items():
        subject_train[indices] = subject
    
    # Subsample BEFORE loading into memory
    if subsample:
        print(f"\n Subsampling {subsample} samples per class...")
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=2*subsample, random_state=42)
        sub_idx, _ = next(sss.split(np.zeros(len(y_train_mmap)), y_train_mmap))
        
        # Load only subsampled indices into memory
        X_train = np.array([X_train_mmap[i] for i in sub_idx])
        y_train = y_train_mmap[sub_idx].copy()
        subject_train = subject_train[sub_idx]
        
        print(f"  ✓ Loaded {len(y_train)} samples into memory")
        print(f"  Classes: {np.bincount(y_train)}")
    else:
        # Load all data into memory
        X_train = np.array(X_train_mmap)
        y_train = np.array(y_train_mmap)
    
    print()
    
    return X_train, y_train, subject_train, manifest


def nested_cv_with_groups(
    X,
    y,
    groups,
    param_grid: Dict,
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
):
    """
    Perform nested grouped cross-validation.
    
    Parameters
    ----------
    X : ndarray
        Feature matrix (time series windows)
    y : ndarray
        Labels
    groups : ndarray
        Subject IDs for grouping
    param_grid : dict
        Parameter grid for hyperparameter search
    n_outer_folds : int
        Number of outer CV folds
    n_inner_folds : int
        Number of inner CV folds
        
    Returns
    -------
    results : dict
        Nested CV results
    """
    outer_cv = GroupKFold(n_splits=n_outer_folds)
    inner_cv = GroupKFold(n_splits=n_inner_folds)
    
    outer_fold_results = []
    best_params_per_fold = []
    best_threshold_per_fold = []
    
    print("\n" + "=" * 80)
    print(f"NESTED GROUPED CV: {n_outer_folds} OUTER & {n_inner_folds} INNER FOLDS")
    print("=" * 80)
    
    # Outer loop: Performance estimation
    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y, groups)):
        print(f"\n{'='*80}")
        print(f"OUTER FOLD {fold_idx + 1}/{n_outer_folds}")
        print(f"{'='*80}")
        
        X_train_outer = X[train_idx]
        y_train_outer = y[train_idx]
        groups_train_outer = groups[train_idx]
        
        X_val_outer = X[val_idx]
        y_val_outer = y[val_idx]
        groups_val_outer = groups[val_idx]
        
        print(f"Train: {len(train_idx)} windows, {len(np.unique(groups_train_outer))} subjects")
        print(f"Val:   {len(val_idx)} windows, {len(np.unique(groups_val_outer))} subjects")
        
        # Verify no subject leakage
        train_subjects = set(groups_train_outer)
        val_subjects = set(groups_val_outer)
        assert len(train_subjects & val_subjects) == 0, "Subject leakage detected in outer fold!"
        
        # Inner loop: Hyperparameter tuning
        print(f"\nInner loop: Tuning hyperparameters...")
        best_score = -np.inf
        best_params = None
        best_threshold = 0.5
        
        param_combinations = list(ParameterGrid(param_grid))
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        for param_idx, params in enumerate(param_combinations):
            inner_scores = []
            inner_thresholds = []
            
            for inner_train_idx, inner_val_idx in inner_cv.split(
                X_train_outer, y_train_outer, groups_train_outer
            ):
                X_inner_train = X_train_outer[inner_train_idx]
                y_inner_train = y_train_outer[inner_train_idx]
                X_inner_val = X_train_outer[inner_val_idx]
                y_inner_val = y_train_outer[inner_val_idx]
                
                # Extract features
                feature_extractor = EnhancedMVARFeatureExtractor(
                    mvar_order=params['mvar_order'],
                    n_basis=params['n_basis'],
                    basis_type=params['basis_type'],
                    regularization=params['regularization'],
                    n_time_points=50,
                    include_per_lag=False,
                )
                
                try:
                    X_inner_train_feats = feature_extractor.transform(X_inner_train, verbose=False)
                    X_inner_val_feats = feature_extractor.transform(X_inner_val, verbose=False)
                except Exception as e:
                    print(f"    Warning: Feature extraction failed: {e}")
                    continue
                
                # Train classifier
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(
                        C=params['C'],
                        class_weight='balanced',
                        max_iter=1000,
                        random_state=42,
                    ))
                ])
                
                # Balanced sample weights for training
                sample_weights = compute_sample_weight(class_weight='balanced', y=y_inner_train)
                pipeline.fit(
                    X_inner_train_feats,
                    y_inner_train,
                    **{'classifier__sample_weight': sample_weights}
                )

                # Evaluate by Average Precision (PR-AUC) on inner validation
                y_inner_proba = pipeline.predict_proba(X_inner_val_feats)[:, 1]
                ap = average_precision_score(y_inner_val, y_inner_proba)
                inner_scores.append(ap)

                # Tune threshold by maximizing F1 on inner validation
                precision, recall, thresholds = precision_recall_curve(y_inner_val, y_inner_proba)
                # Compute F1 for points where both precision and recall are defined
                denom = (precision + recall)
                f1 = np.where(denom > 0, 2 * precision * recall / denom, 0)
                # thresholds length is len(precision)-1; align by skipping last f1
                if thresholds.size > 0 and f1.size > 1:
                    best_idx = int(np.nanargmax(f1[:-1]))
                    inner_thresholds.append(float(thresholds[best_idx]))
                else:
                    inner_thresholds.append(0.5)
            
            mean_score = np.mean(inner_scores)
            median_threshold = float(np.median(inner_thresholds))
            
            if (param_idx + 1) % 5 == 0:
                print(f"  Tested {param_idx + 1}/{len(param_combinations)} combinations...")
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_threshold = median_threshold
        
        print(f"\n✓ Best params for fold {fold_idx + 1}: {best_params}")
        print(f"  Inner CV score: {best_score:.4f}")
        print(f"  Selected decision threshold (median inner): {best_threshold:.3f}")
        best_params_per_fold.append(best_params)
        best_threshold_per_fold.append(best_threshold)
        
        # Retrain on full outer training set with best params
        print(f"\nRetraining on full outer training set...")
        feature_extractor = EnhancedMVARFeatureExtractor(
            mvar_order=best_params['mvar_order'],
            n_basis=best_params['n_basis'],
            basis_type=best_params['basis_type'],
            regularization=best_params['regularization'],
            n_time_points=50,
            include_per_lag=False,
        )
        
        X_train_feats = feature_extractor.transform(X_train_outer, verbose=False)
        X_val_feats = feature_extractor.transform(X_val_outer, verbose=False)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=best_params['C'],
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
            ))
        ])
        
        # Balanced sample weights on full outer training set
        outer_sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_outer)
        pipeline.fit(
            X_train_feats,
            y_train_outer,
            **{'classifier__sample_weight': outer_sample_weights}
        )
        
        # Evaluate on outer validation set
        y_proba_pos = pipeline.predict_proba(X_val_feats)[:, 1]
        y_pred = (y_proba_pos >= best_threshold).astype(int)
        
        fold_metrics = compute_comprehensive_metrics(y_val_outer, y_pred, y_proba_pos)
        outer_fold_results.append(fold_metrics)
        
        print(f"\nOuter fold {fold_idx + 1} results:")
        print(f"  Accuracy: {fold_metrics['accuracy']:.4f}")
        print(f"  ROC-AUC:  {fold_metrics.get('roc_auc', np.nan):.4f}")
        print(f"  Sensitivity: {fold_metrics['sensitivity']:.4f}")
        print(f"  Specificity: {fold_metrics['specificity']:.4f}")
    
    # Aggregate results
    print("\n" + "=" * 80)
    print("NESTED CV RESULTS SUMMARY")
    print("=" * 80)
    
    # Compute mean and std across folds
    metrics_summary = {}
    for metric in outer_fold_results[0].keys():
        values = [fold[metric] for fold in outer_fold_results if not np.isnan(fold[metric])]
        if len(values) > 0:
            metrics_summary[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'folds': [float(v) for v in values],
            }
    
    print("\nPerformance across folds:")
    print(f"  Accuracy:     {metrics_summary['accuracy']['mean']:.4f} ± {metrics_summary['accuracy']['std']:.4f}")
    if 'roc_auc' in metrics_summary:
        print(f"  ROC-AUC:      {metrics_summary['roc_auc']['mean']:.4f} ± {metrics_summary['roc_auc']['std']:.4f}")
    print(f"  Sensitivity:  {metrics_summary['sensitivity']['mean']:.4f} ± {metrics_summary['sensitivity']['std']:.4f}")
    print(f"  Specificity:  {metrics_summary['specificity']['mean']:.4f} ± {metrics_summary['specificity']['std']:.4f}")
    
    print("\nBest parameters per fold:")
    for i, params in enumerate(best_params_per_fold):
        print(f"  Fold {i+1}: {params}")
    print("\nDecision thresholds per fold:")
    for i, thr in enumerate(best_threshold_per_fold):
        print(f"  Fold {i+1}: {thr:.3f}")
    
    results = {
        'metrics_summary': metrics_summary,
        'fold_results': outer_fold_results,
        'best_params_per_fold': best_params_per_fold,
        'best_threshold_per_fold': best_threshold_per_fold,
        'n_outer_folds': n_outer_folds,
        'n_inner_folds': n_inner_folds,
    }
    
    return results


def main():
    """Main nested CV pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-outer-folds', type=int, default=3, help='Number of outer CV folds (default: 3 for speed)')
    parser.add_argument('--n-inner-folds', type=int, default=2, help='Number of inner CV folds (default: 2 for speed)')
    parser.add_argument('--output-dir', type=Path, default=PATHS['results'] / 'nested_cv', help='Output directory')
    parser.add_argument('--subsample', type=int, default=None, help='Subsample N samples per class')
    parser.add_argument('--fast-mode', action='store_true', help='Use minimal parameter grid for quick testing')
    args = parser.parse_args()
    
    # Load data (with subsampling applied during load)
    X, y, groups, manifest = load_data(subsample=args.subsample)
    
    # Define parameter grid
    if args.fast_mode:
        print("\n⚡ FAST MODE: Using minimal parameter grid")
        param_grid = {
            'mvar_order': [3],
            'n_basis': [8],
            'basis_type': ['bspline'],
            'regularization': [0.01],
            'C': [1.0],
        }
    else:
        param_grid = {
            'mvar_order': [3, 5],
            'n_basis': [8, 10],
            'basis_type': ['bspline'],
            'regularization': [0.01, 0.1],
            'C': [1.0, 10.0],
        }
    
    print("\nParameter grid:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    print(f"\nTotal combinations: {np.prod([len(v) for v in param_grid.values()])}")
    
    # Run nested CV
    results = nested_cv_with_groups(
        X=X,
        y=y,
        groups=groups,
        param_grid=param_grid,
        n_outer_folds=args.n_outer_folds,
        n_inner_folds=args.n_inner_folds,
    )
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(args.output_dir / 'nested_cv_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output_dir}")
    print("\n" + "=" * 80)
    print("✓ NESTED CV COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
