"""
Diagnostic script to analyze seizure detection performance issues.

Run this to understand why your model has poor seizure detection.
"""

import json
import numpy as np
from pathlib import Path

# Load manifest
with open('artifacts/manifest.json', 'r') as f:
    manifest = json.load(f)

# Load data
y_train = np.memmap('artifacts/y_train.npy', dtype=np.int8, mode='r', shape=(manifest['n_train'],))
y_test = np.memmap('artifacts/y_test.npy', dtype=np.int8, mode='r', shape=(manifest['n_test'],))

# Load results if available
results_path = Path('src/results/evaluation_results.json')
if results_path.exists():
    with open(results_path, 'r') as f:
        results = json.load(f)
else:
    results = None

print("=" * 80)
print("SEIZURE DETECTION DIAGNOSTIC REPORT")
print("=" * 80)

# 1. Dataset imbalance
print("\n1. DATASET CLASS DISTRIBUTION")
print("-" * 80)
print(f"Train set:")
print(f"  Total: {len(y_train):,}")
train_seizure_pct = 100 * manifest['train_class_1'] / manifest['n_train']
print(f"  Non-seizure: {manifest['train_class_0']:,} ({100-train_seizure_pct:.2f}%)")
print(f"  Seizure:     {manifest['train_class_1']:,} ({train_seizure_pct:.2f}%)")
print(f"  Imbalance ratio: 1:{manifest['train_class_0']/manifest['train_class_1']:.1f}")

print(f"\nTest set:")
print(f"  Total: {len(y_test):,}")
test_seizure_pct = 100 * manifest['test_class_1'] / manifest['n_test']
print(f"  Non-seizure: {manifest['test_class_0']:,} ({100-test_seizure_pct:.2f}%)")
print(f"  Seizure:     {manifest['test_class_1']:,} ({test_seizure_pct:.2f}%)")
print(f"  Imbalance ratio: 1:{manifest['test_class_0']/manifest['test_class_1']:.1f}")

print(f"\n⚠️  TEST SET IS {(manifest['test_class_0']/manifest['test_class_1']) / (manifest['train_class_0']/manifest['train_class_1']):.1f}x MORE IMBALANCED THAN TRAIN!")

# 2. Per-subject analysis
print("\n2. PER-SUBJECT SEIZURE DISTRIBUTION")
print("-" * 80)
print("Test subjects:")
for subject, indices in manifest['subject_window_map_test'].items():
    subject_labels = y_test[indices]
    n_seizure = np.sum(subject_labels)
    n_total = len(subject_labels)
    print(f"  {subject}: {n_seizure:,} seizures / {n_total:,} windows ({100*n_seizure/n_total:.2f}%)")

print("\nTrain subjects (top 5 by seizure count):")
train_subject_seizures = []
for subject, indices in manifest['subject_window_map_train'].items():
    subject_labels = y_train[indices]
    n_seizure = np.sum(subject_labels)
    n_total = len(subject_labels)
    train_subject_seizures.append((subject, n_seizure, n_total))

train_subject_seizures.sort(key=lambda x: x[1], reverse=True)
for subject, n_seizure, n_total in train_subject_seizures[:5]:
    print(f"  {subject}: {n_seizure:,} seizures / {n_total:,} windows ({100*n_seizure/n_total:.2f}%)")

# 3. Subsampling impact
print("\n3. SUBSAMPLING IMPACT (common settings)")
print("-" * 80)
from sklearn.model_selection import StratifiedShuffleSplit

for subsample in [100, 500, 1000, 2000]:
    sss = StratifiedShuffleSplit(n_splits=1, train_size=2*subsample, random_state=42)
    test_sub_idx, _ = next(sss.split(np.zeros(len(y_test)), y_test))
    y_test_sub = y_test[test_sub_idx]
    test_counts = np.bincount(y_test_sub)
    
    if len(test_counts) > 1:
        n_seizure = test_counts[1]
    else:
        n_seizure = 0
    
    print(f"--subsample {subsample}:")
    print(f"  Test samples: {len(y_test_sub)}")
    print(f"  Seizure samples: {n_seizure} ({100*n_seizure/len(y_test_sub):.2f}%)")
    if n_seizure < 30:
        print(f"  ⚠️  WARNING: Only {n_seizure} seizure samples - too few for reliable evaluation!")

# 4. Model performance (if results exist)
if results:
    print("\n4. MODEL PERFORMANCE ANALYSIS")
    print("-" * 80)
    wl = results['window_level']
    
    print(f"Window-level metrics:")
    print(f"  Accuracy: {wl['accuracy']:.3f}")
    print(f"  Sensitivity (recall): {wl['sensitivity']:.3f}")
    print(f"  Specificity: {wl['specificity']:.3f}")
    print(f"  Precision (seizure): {wl['precision_seizure']:.3f}")
    print(f"  F1 (seizure): {wl['f1_seizure']:.3f}")
    print(f"  ROC-AUC: {wl.get('roc_auc', 'N/A')}")
    print(f"  PR-AUC: {wl.get('pr_auc', 'N/A')}")
    
    print("\n⚠️  DIAGNOSIS:")
    if wl['sensitivity'] > 0.5 and wl['precision_seizure'] < 0.05:
        print("  - Model detects seizures (sensitivity > 50%)")
        print("  - But has VERY low precision (< 5%)")
        print("  - This means: Lots of FALSE POSITIVES")
        print("  - Root cause: Extreme class imbalance in test set")
    
    if wl.get('roc_auc', 0.5) < 0.5:
        print("  - ROC-AUC < 0.5 suggests model is performing worse than random")
        print("  - This is likely due to poor calibration on imbalanced test set")
    
    sl = results.get('subject_level_majority', {})
    if sl.get('sensitivity', 0) == 0:
        print("  - Subject-level sensitivity = 0%")
        print("  - Model doesn't correctly classify ANY test subjects")
        print("  - Likely due to majority-vote being biased by abundant non-seizure windows")

# 5. Recommendations
print("\n5. RECOMMENDED FIXES")
print("-" * 80)
print("✓ Immediate:")
print("  1. Don't subsample test set - use ALL 282 seizure samples")
print("  2. Tune decision threshold (try 0.3 or lower instead of 0.5)")
print("  3. Focus on PR-AUC and F2-score instead of accuracy")

print("\n✓ Data preprocessing:")
print("  4. Increase test_size to 0.4 for better subject distribution")
print("  5. Implement stratified subject split to balance seizure rates")

print("\n✓ Model improvements:")
print("  6. Use focal loss or cost-sensitive learning")
print("  7. Add temporal context (e.g., neighboring windows)")
print("  8. Try different classification thresholds per subject")

print("\n✓ Evaluation:")
print("  9. Report sensitivity at fixed specificity (e.g., 90%)")
print("  10. Use time-aware metrics (seizure detection rate, false alarm rate per hour)")

print("\n" + "=" * 80)
print("See SEIZURE_DETECTION_ISSUES.md for detailed solutions")
print("=" * 80)
