#python scripts/00_preprocessing.py

from __future__ import annotations

import argparse
import logging
import os
import pickle
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import json
from datetime import datetime
from tqdm import tqdm
import mne
import numpy as np
import pandas as pd


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.parameters import DEFAULT_PARAMS
from src.data.paths import PATHS
from src.data.splits import (
    create_subject_grouped_split, 
    create_stratified_subject_split,
    print_split_statistics
)

# Suppress MNE warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

TIME_RE = re.compile(r"(\d{1,2}\.\d{1,2}\.\d{1,2})")
TIME_ALT_RE = re.compile(r"(\d{1,2}):(\d{1,2})\.(\d{1,2})")


# ---------- Logging setup ----------
@dataclass
class LoadErrorTracker:
    """Track errors during data loading for diagnostics."""
    total_files: int = 0
    successful_files: int = 0
    failed_files: Dict[str, str] = field(default_factory=dict)  # path -> error reason
    failed_subjects: Dict[str, List[str]] = field(default_factory=dict)  # subject -> [error_reasons]
    
    def add_error(self, file_path: str | Path, subject: str, reason: str):
        """Record a failed file."""
        file_path = str(file_path)
        self.failed_files[file_path] = reason
        if subject not in self.failed_subjects:
            self.failed_subjects[subject] = []
        self.failed_subjects[subject].append(f"{Path(file_path).name}: {reason}")
    
    def report(self) -> str:
        """Generate error summary report."""
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("DATA LOADING ERROR SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Total files examined: {self.total_files}")
        lines.append(f"Successfully loaded: {self.successful_files}")
        lines.append(f"Failed: {len(self.failed_files)}")
        
        if self.failed_files:
            lines.append("\nFailed subjects:")
            for subject in sorted(self.failed_subjects.keys()):
                errors = self.failed_subjects[subject]
                lines.append(f"\n  {subject} ({len(errors)} file(s)):")
                for error in errors:
                    lines.append(f"    - {error}")
        
        lines.append("=" * 80)
        return "\n".join(lines)


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to file and console."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger = logging.getLogger("preprocessing")
    logger.setLevel(logging.DEBUG)
    
    # File handler (DEBUG level)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Console handler (INFO level)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"Logging initialized. Detailed logs: {log_file}")
    
    return logger


# ---------- Data structures ----------
@dataclass
class SeizureInterval:
    """Seizure interval metadata."""
    file_name: str
    reg_start_s: float
    reg_end_s: float
    seizure_start_s: float
    seizure_end_s: float
    sample_rate: float


# ---------- Time parsing utilities ----------
def _parse_time_seconds(line: str) -> float:
    """Parse time from line in format HH.MM.SS or HH:MM.SS"""
    # Try standard format first (HH.MM.SS)
    match = TIME_RE.search(line.replace(" ", ""))
    if match:
        hours, minutes, seconds = map(int, match.group(1).split("."))
        return hours * 3600 + minutes * 60 + seconds
    
    # Try alternative format (HH:MM.SS)
    match = TIME_ALT_RE.search(line.replace(" ", ""))
    if match:
        hours, minutes, seconds = map(int, match.groups())
        return hours * 3600 + minutes * 60 + seconds
    
    raise ValueError(f"Unable to parse time from: {line}")


def _diff_with_wrap(later: float, earlier: float) -> float:
    """Handle time differences that may wrap across 24h boundary."""
    delta = later - earlier
    return delta if delta >= 0 else delta + 24 * 3600


# ---------- Metadata parsing ----------
def parse_seizure_list(txt_path: Path) -> List[SeizureInterval]:
    """Parse seizure text file and extract intervals."""
    lines = txt_path.read_text().splitlines()
    sr = 512.0
    blocks: List[SeizureInterval] = []
    pending: Dict[str, float | str | None] = {
        "file_name": None,
        "reg_start": None,
        "reg_end": None,
        "seiz_start": None,
        "seiz_end": None,
    }

    def flush():
        """Flush a completed seizure interval."""
        if all(pending[k] is not None for k in ("file_name", "reg_start", "reg_end", "seiz_start", "seiz_end")):
            reg_start = float(pending["reg_start"])
            reg_end = float(pending["reg_end"])
            seizure_start = float(pending["seiz_start"])
            seizure_end = float(pending["seiz_end"])
            blocks.append(
                SeizureInterval(
                    file_name=str(pending["file_name"]),
                    reg_start_s=reg_start,
                    reg_end_s=_diff_with_wrap(reg_end, reg_start),
                    seizure_start_s=_diff_with_wrap(seizure_start, reg_start),
                    seizure_end_s=_diff_with_wrap(seizure_end, reg_start),
                    sample_rate=sr,
                )
            )
            pending.update({
                "file_name": None,
                "reg_start": None,
                "reg_end": None,
                "seiz_start": None,
                "seiz_end": None,
            })

    for line in lines:
        if not line.strip():
            continue
        if "Sampling Rate" in line:
            sr_match = re.search(r"([0-9]+)\s*Hz", line)
            if sr_match:
                sr = float(sr_match.group(1))
            continue
        if "File name" in line:
            flush()
            pending["file_name"] = line.split(":", 1)[1].strip()
            continue
        if "Registration start" in line:
            try:
                pending["reg_start"] = _parse_time_seconds(line)
            except ValueError:
                pass
            continue
        if "Registration end" in line:
            try:
                pending["reg_end"] = _parse_time_seconds(line)
            except ValueError:
                pass
            continue
        if "Seizure start" in line:
            try:
                pending["seiz_start"] = _parse_time_seconds(line)
            except ValueError:
                pass
            continue
        if "Seizure end" in line:
            try:
                pending["seiz_end"] = _parse_time_seconds(line)
            except ValueError:
                pass
            flush()

    flush()
    return blocks


def load_valid_subjects(dataset_root: Path, logger: logging.Logger, required_channels: int = 29) -> set:
    """
    Load subject IDs that have the required number of EEG channels.
    
    Parameters
    ----------
    dataset_root : Path
        Root directory of the dataset
    logger : logging.Logger
        Logger instance
    required_channels : int
        Required number of EEG channels (default: 29)
        
    Returns
    -------
    valid_subjects : set
        Set of subject IDs with exactly the required number of channels
    """
    subject_info_path = dataset_root / "subject_info.csv"
    
    if not subject_info_path.exists():
        logger.warning(f"subject_info.csv not found at {subject_info_path}. All subjects will be processed.")
        return None
    
    try:
        df = pd.read_csv(subject_info_path)
        # Clean column names (remove spaces)
        df.columns = df.columns.str.strip()
        
        # Filter subjects with exactly the required number of channels
        valid_df = df[df['eeg_channel'] == required_channels]
        valid_subjects = set(valid_df['patient_id'].astype(str).str.strip())
        
        excluded_subjects = set(df['patient_id'].astype(str).str.strip()) - valid_subjects
        
        logger.info(f"Loaded subject info from {subject_info_path}")
        logger.info(f"Valid subjects with {required_channels} channels: {len(valid_subjects)}")
        logger.info(f"Valid subjects: {sorted(valid_subjects)}")
        if excluded_subjects:
            logger.info(f"Excluded subjects (≠{required_channels} channels): {sorted(excluded_subjects)}")
        
        return valid_subjects
    except Exception as e:
        logger.error(f"Failed to load subject_info.csv: {e}")
        raise


def collect_metadata(dataset_root: Path, logger: logging.Logger, error_tracker: LoadErrorTracker, 
                     valid_subjects: set = None) -> Dict[str, Dict[str, object]]:
    """
    Collect metadata from all seizure list files.
    
    Parameters
    ----------
    dataset_root : Path
        Root directory of the dataset
    logger : logging.Logger
        Logger instance
    error_tracker : LoadErrorTracker
        Error tracking instance
    valid_subjects : set, optional
        Set of valid subject IDs to include. If None, all subjects are included.
    
    Returns
    -------
    metadata : dict
        Dictionary mapping file keys to metadata
    """
    metadata: Dict[str, Dict[str, object]] = {}
    seizure_files = list(dataset_root.glob("PN*/Seizures-list-*.txt"))
    
    if not seizure_files:
        logger.warning(f"No seizure list files found in {dataset_root}")
        return metadata
    
    logger.info(f"Found {len(seizure_files)} seizure list files")
    
    for seizures_txt in seizure_files:
        subject = seizures_txt.parent.name
        
        # Skip subjects not in valid_subjects set (if provided)
        if valid_subjects is not None and subject not in valid_subjects:
            logger.debug(f"Skipping {subject}: not in valid subjects list")
            continue
        try:
            for interval in parse_seizure_list(seizures_txt):
                key = f"{subject}/{interval.file_name}"
                if key not in metadata:
                    metadata[key] = {
                        "subject": subject,
                        "file_name": interval.file_name,
                        "reg_end": interval.reg_end_s,
                        "seizures": [],
                        "sample_rate": interval.sample_rate,
                    }
                metadata[key]["seizures"].append((interval.seizure_start_s, interval.seizure_end_s))
            logger.debug(f"Parsed metadata for {subject}: {seizures_txt.name}")
        except Exception as e:
            logger.error(f"Failed to parse {seizures_txt}: {e}")
            error_tracker.add_error(seizures_txt, subject, f"Metadata parse error: {e}")
    
    logger.info(f"Collected metadata for {len(metadata)} file(s)")
    return metadata


# ---------- Window extraction ----------
def _non_seizure_intervals(
    seizures: Sequence[Tuple[float, float]], reg_end: float, window_sec: float
) -> List[Tuple[float, float]]:
    """Extract non-seizure intervals that have sufficient duration."""
    spans: List[Tuple[float, float]] = []
    cur = 0.0
    for start, end in sorted(seizures):
        if start - cur >= window_sec:
            spans.append((cur, start))
        cur = max(cur, end)
    if reg_end - cur >= window_sec:
        spans.append((cur, reg_end))
    return spans


def _count_interval_windows(
    intervals: Iterable[Tuple[float, float]],
    sr: float,
    win_samples: int,
    stride_samples: int,
) -> int:
    """Count how many windows of size win_samples fit in the intervals (no IO)."""
    total = 0
    for start_s, end_s in intervals:
        start = int(round(start_s * sr))
        end = int(round(end_s * sr))
        length = end - start
        if length >= win_samples:
            n = 1 + (length - win_samples) // stride_samples
            total += int(n)
    return total


def _stream_windows_from_raw(
    raw: mne.io.BaseRaw,
    picks: np.ndarray,
    intervals: Iterable[Tuple[float, float]],
    sr: float,
    win_samples: int,
    stride_samples: int,
):
    """Yield windows one-by-one without loading whole recording into memory."""
    for start_s, end_s in intervals:
        s_start = int(round(start_s * sr))
        s_end = int(round(end_s * sr))
        if s_end - s_start < win_samples:
            continue
        for s in range(s_start, s_end - win_samples + 1, stride_samples):
            # Read only the needed chunk
            chunk = raw.get_data(picks=picks, start=s, stop=s + win_samples)
            # Ensure float32 to reduce RAM
            yield chunk.astype(np.float32, copy=False)


# ---------- Main preprocessing ----------
def build_balanced_split(
    dataset_root: Path,
    window_sec: float,
    stride_sec: float,
    test_size: float,
    random_state: int,
    output_dir: Path,
    use_stratified_split: bool = True,
    balance_strategy: str = "none",
    balance_test: bool = False,
    logger: logging.Logger | None = None,
) -> Path:
    """Preprocessing with subject-grouped stratified split (prevents leakage).

    Streaming, two-pass design to avoid out-of-memory:
      - Pass 1: subject-level split + count windows (train/test)
      - Pass 2: stream windows and write directly to memmaps

    Parameters
    ----------
    use_stratified_split : bool
        If True, use stratified subject split to balance seizure distribution.
        If False, use random subject split (may have severe imbalance).
        
    Preserves original style: banners, args, and outputs.
    """
    if logger is None:
        logger = logging.getLogger("preprocessing")
    
    error_tracker = LoadErrorTracker()
    
    # Load valid subjects (those with exactly 29 electrodes)
    valid_subjects = load_valid_subjects(dataset_root, logger, required_channels=29)
    
    metadata = collect_metadata(dataset_root, logger, error_tracker, valid_subjects=valid_subjects)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate: float | None = None
    n_channels: int | None = None

    # ---------- Subject list and compute per-subject seizure rates ----------
    subjects = sorted({meta["subject"] for meta in metadata.values()})
    logger.info(f"Found {len(subjects)} unique subjects in metadata")
    
    if not subjects:
        logger.error("No subjects found in metadata. Cannot proceed.")
        raise ValueError("No valid subjects in metadata")
    
    # Compute actual seizure rates per subject from metadata
    subject_seizure_info: Dict[str, Dict] = {}
    for subject in subjects:
        total_windows = 0
        seizure_windows = 0
        
        for key, meta in metadata.items():
            if meta["subject"] != subject:
                continue
            
            # Count windows for this file (approximate from metadata)
            seizures = meta.get("seizures", [])
            reg_end = meta.get("reg_end", 0)
            
            # Estimate window counts (will be exact in pass 1)
            for sz_start, sz_end in seizures:
                sz_dur = sz_end - sz_start
                seizure_windows += int(sz_dur / window_sec)
            
            # Approximate total duration
            total_dur = reg_end
            if seizures:
                non_sz_dur = total_dur - sum(end - start for start, end in seizures)
                total_windows += int(non_sz_dur / window_sec)
            else:
                total_windows += int(total_dur / window_sec)
            total_windows += seizure_windows
        
        seizure_rate = seizure_windows / total_windows if total_windows > 0 else 0
        subject_seizure_info[subject] = {
            'seizure_rate': seizure_rate,
            'approx_seizure_windows': seizure_windows,
            'approx_total_windows': total_windows
        }
    
    # Log subject seizure rates
    logger.info("\nPer-subject seizure rates (approximate):")
    for subject in sorted(subjects, key=lambda s: subject_seizure_info[s]['seizure_rate'], reverse=True):
        info = subject_seizure_info[subject]
        logger.info(f"  {subject}: {info['approx_seizure_windows']}/{info['approx_total_windows']} "
                   f"({100*info['seizure_rate']:.2f}% seizure)")
    
    # Choose split method
    if use_stratified_split:
        logger.info("\n✓ Using STRATIFIED subject split to balance seizure distribution")
        print("\n✓ Using STRATIFIED subject split to balance seizure distribution")
        
        # Sort subjects by seizure rate and alternate assignment
        sorted_subjects = sorted(subjects, key=lambda s: subject_seizure_info[s]['seizure_rate'])
        n_test = max(1, int(len(subjects) * test_size))
        
        # Distribute evenly across sorted list
        test_subjects_list = []
        step = len(sorted_subjects) / n_test
        for i in range(n_test):
            idx = int(i * step)
            if idx < len(sorted_subjects):
                test_subjects_list.append(sorted_subjects[idx])
        
        train_subjects = set(s for s in subjects if s not in test_subjects_list)
        test_subjects = set(test_subjects_list)
        
    else:
        logger.info("\n⚠️  Using RANDOM subject split (may have severe imbalance)")
        print("\n⚠️  Using RANDOM subject split (may have severe imbalance)")
        
        from sklearn.model_selection import StratifiedShuffleSplit
        # Stratify by whether subject has any seizures
        subj_has_seizure = {s: 1 if subject_seizure_info[s]['seizure_rate'] > 0 else 0 
                           for s in subjects}
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        subj_labels = np.array([subj_has_seizure[s] for s in subjects])
        train_subj_idx, test_subj_idx = next(sss.split(np.zeros(len(subjects)), subj_labels))
        train_subjects = set(np.array(subjects)[train_subj_idx])
        test_subjects = set(np.array(subjects)[test_subj_idx])
    
    logger.info(f"\nTrain subjects ({len(train_subjects)}): {sorted(train_subjects)}")
    logger.info(f"Test subjects ({len(test_subjects)}): {sorted(test_subjects)}")
    
    # Log expected seizure rates for splits
    train_sz_rate = np.mean([subject_seizure_info[s]['seizure_rate'] for s in train_subjects])
    test_sz_rate = np.mean([subject_seizure_info[s]['seizure_rate'] for s in test_subjects])
    logger.info(f"\nExpected mean seizure rates:")
    logger.info(f"  Train: {100*train_sz_rate:.2f}%")
    logger.info(f"  Test:  {100*test_sz_rate:.2f}%")
    
    if test_sz_rate < train_sz_rate * 0.5:
        logger.warning(f"  ⚠️  Test seizure rate is < 50% of train rate - may have severe imbalance!")
    else:
        logger.info(f"  ✓ Split appears balanced")

    print("\n" + "=" * 80)
    print("PHASE 1: SUBJECT SPLIT + WINDOW COUNTING (NO DATA LOAD)")
    print("=" * 80)

    # First pass: verify sr/channels and count windows per split
    train_counts = {"non": 0, "sz": 0}
    test_counts = {"non": 0, "sz": 0}
    first_win_shape = None
    
    # Track channel counts to find most common
    channel_counts: Dict[int, int] = {}

    for key, meta in tqdm(metadata.items(), desc="Counting windows"):
        subject = meta["subject"]
        edf_name = meta["file_name"]
        edf_path = dataset_root / subject / edf_name
        
        error_tracker.total_files += 1
        
        if not edf_path.exists():
            reason = f"File not found at {edf_path}"
            logger.warning(reason)
            error_tracker.add_error(edf_path, subject, reason)
            continue
        
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            sr = float(raw.info["sfreq"])  
            picks = mne.pick_types(raw.info, eeg=True)
            
            if len(picks) == 0:
                reason = f"No EEG channels found (got {len(picks)})"
                logger.debug(f"{edf_path}: {reason}")
                error_tracker.add_error(edf_path, subject, reason)
                del raw
                continue
            
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                reason = f"Sampling rate mismatch: expected {sample_rate} Hz, got {sr} Hz"
                logger.debug(f"{edf_path}: {reason}")
                error_tracker.add_error(edf_path, subject, reason)
                del raw
                continue
            
            # Track channel counts (don't reject yet)
            n_ch = int(len(picks))
            channel_counts[n_ch] = channel_counts.get(n_ch, 0) + 1

            win_samples = int(round(window_sec * sr))
            stride_samples = int(round(stride_sec * sr))
            seizures = meta["seizures"]
            reg_end = float(meta["reg_end"])
            non_seizures = _non_seizure_intervals(seizures, reg_end, window_sec)

            n_non = _count_interval_windows(non_seizures, sr, win_samples, stride_samples)
            n_sz = _count_interval_windows(seizures, sr, win_samples, stride_samples)

            if subject in train_subjects:
                train_counts["non"] += n_non
                train_counts["sz"] += n_sz
            else:
                test_counts["non"] += n_non
                test_counts["sz"] += n_sz

            logger.debug(f"✓ {edf_path.name}: {n_non} non-seizure + {n_sz} seizure windows ({n_ch} channels)")
            error_tracker.successful_files += 1
            del raw
        except Exception as e:
            reason = f"Read error: {type(e).__name__}: {str(e)[:100]}"
            logger.warning(f"{edf_path}: {reason}")
            error_tracker.add_error(edf_path, subject, reason)
            continue
    
    # Determine canonical channel count (most common)
    if channel_counts:
        n_channels = max(channel_counts, key=channel_counts.get)
        logger.info(f"Channel count distribution: {dict(sorted(channel_counts.items()))}")
        logger.info(f"Using most common channel count: {n_channels} ({channel_counts[n_channels]} files)")
    
    if sample_rate is None or n_channels is None:
        logger.error("No valid EDF files found with EEG channels.")
        raise ValueError("No valid EDF files found with EEG channels.")
    
    # Re-scan once to get window shape with correct n_channels
    win_samples = None
    for key, meta in metadata.items():
        subject = meta["subject"]
        edf_name = meta["file_name"]
        edf_path = dataset_root / subject / edf_name
        if not edf_path.exists():
            continue
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            sr = float(raw.info["sfreq"])
            if sr == sample_rate:
                win_samples = int(round(window_sec * sr))
                first_win_shape = (n_channels, win_samples)
                del raw
                break
            del raw
        except:
            continue
    
    if first_win_shape is None:
        logger.error("No valid EDF files found with EEG channels.")
        raise ValueError("No valid EDF files found with EEG channels.")

    n_train_total = train_counts["non"] + train_counts["sz"]
    n_test_total = test_counts["non"] + test_counts["sz"]

    logger.info(f"Train windows: non={train_counts['non']}, sz={train_counts['sz']} (total {n_train_total})")
    logger.info(f"Test windows:  non={test_counts['non']}, sz={test_counts['sz']} (total {n_test_total})")

    print(f"Train windows: non={train_counts['non']}, sz={train_counts['sz']} (total {n_train_total})")
    print(f"Test windows:  non={test_counts['non']}, sz={test_counts['sz']} (total {n_test_total})")

    # ---------- Allocate memmaps ----------
    X_train_path = output_dir / "X_train.npy"
    X_test_path = output_dir / "X_test.npy"
    y_train_path = output_dir / "y_train.npy"
    y_test_path = output_dir / "y_test.npy"
    subject_train_path = output_dir / "subject_train.txt"
    subject_test_path = output_dir / "subject_test.txt"

    X_train = np.memmap(X_train_path, mode="w+", dtype=np.float32, shape=(n_train_total, first_win_shape[0], first_win_shape[1]))
    X_test = np.memmap(X_test_path, mode="w+", dtype=np.float32, shape=(n_test_total, first_win_shape[0], first_win_shape[1]))
    y_train = np.memmap(y_train_path, mode="w+", dtype=np.int8, shape=(n_train_total,))
    y_test = np.memmap(y_test_path, mode="w+", dtype=np.int8, shape=(n_test_total,))

    # ---------- Second pass: stream and write ----------
    print("\n" + "=" * 80)
    print("PHASE 2: STREAMING WINDOWS AND WRITING TO MEMMAPS")
    print("=" * 80)

    train_cursor = 0
    test_cursor = 0
    train_written = {"non": 0, "sz": 0}
    test_written = {"non": 0, "sz": 0}

    # Track subject-to-window-index mapping instead of storing subjects per window
    subject_window_map_train: Dict[str, List[int]] = {s: [] for s in train_subjects}
    subject_window_map_test: Dict[str, List[int]] = {s: [] for s in test_subjects}

    try:
        for key, meta in tqdm(metadata.items(), desc="Streaming windows"):
            subject = meta["subject"]
            edf_name = meta["file_name"]
            edf_path = dataset_root / subject / edf_name
            if not edf_path.exists():
                continue
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
                sr = float(raw.info["sfreq"])  
                if sr != sample_rate:
                    continue
                picks = mne.pick_types(raw.info, eeg=True)
                
                # Handle variable channel counts by selecting or padding
                actual_channels = len(picks)
                if actual_channels != n_channels:
                    if actual_channels < n_channels:
                        # Pad with zeros for missing channels
                        logger.debug(f"{edf_path.name}: Padding {n_channels - actual_channels} channels")
                    else:
                        # Select first n_channels
                        logger.debug(f"{edf_path.name}: Selecting first {n_channels} of {actual_channels} channels")
                        picks = picks[:n_channels]

                win_samples = int(round(window_sec * sr))
                stride_samples = int(round(stride_sec * sr))
                seizures = meta["seizures"]
                reg_end = float(meta["reg_end"])
                non_seizures = _non_seizure_intervals(seizures, reg_end, window_sec)

                is_train = subject in train_subjects

                # Non-seizure first
                for w in _stream_windows_from_raw(raw, picks, non_seizures, sr, win_samples, stride_samples):
                    # Pad if needed
                    if w.shape[0] < n_channels:
                        w_padded = np.zeros((n_channels, win_samples), dtype=np.float32)
                        w_padded[:w.shape[0]] = w
                        w = w_padded
                    
                    if is_train:
                        X_train[train_cursor] = w
                        y_train[train_cursor] = 0
                        subject_window_map_train[subject].append(train_cursor)
                        train_cursor += 1
                        train_written["non"] += 1
                    else:
                        X_test[test_cursor] = w
                        y_test[test_cursor] = 0
                        subject_window_map_test[subject].append(test_cursor)
                        test_cursor += 1
                        test_written["non"] += 1

                # Seizure next
                for w in _stream_windows_from_raw(raw, picks, seizures, sr, win_samples, stride_samples):
                    # Pad if needed
                    if w.shape[0] < n_channels:
                        w_padded = np.zeros((n_channels, win_samples), dtype=np.float32)
                        w_padded[:w.shape[0]] = w
                        w = w_padded
                    
                    if is_train:
                        X_train[train_cursor] = w
                        y_train[train_cursor] = 1
                        subject_window_map_train[subject].append(train_cursor)
                        train_cursor += 1
                        train_written["sz"] += 1
                    else:
                        X_test[test_cursor] = w
                        y_test[test_cursor] = 1
                        subject_window_map_test[subject].append(test_cursor)
                        test_cursor += 1
                        test_written["sz"] += 1

                del raw
            except Exception as e:
                reason = f"Stream error: {type(e).__name__}: {str(e)[:100]}"
                logger.debug(f"{edf_path}: {reason}")
                error_tracker.add_error(edf_path, subject, reason)
                continue
    finally:
        pass

    # Flush memmaps
    del X_train, X_test, y_train, y_test

    print("\n✓ Data written to disk")
    print(f"Train written: non={train_written['non']}, sz={train_written['sz']} (total {train_cursor})")
    print(f"Test written:  non={test_written['non']}, sz={test_written['sz']} (total {test_cursor})")
    
    logger.info(f"Train written: non={train_written['non']}, sz={train_written['sz']} (total {train_cursor})")
    logger.info(f"Test written:  non={test_written['non']}, sz={test_written['sz']} (total {test_cursor})")

    # Optionally balance classes (overwrite train/test artifacts)
    if balance_strategy not in ("none", None):
        import shutil

        np.random.seed(random_state)

        # Balance training set
        if train_written['non'] > 0 and train_written['sz'] > 0:
            if balance_strategy == 'undersample':
                target_train_per_class = min(train_written['non'], train_written['sz'])
            elif balance_strategy == 'oversample':
                target_train_per_class = max(train_written['non'], train_written['sz'])
            else:
                target_train_per_class = min(train_written['non'], train_written['sz'])

            # Paths
            X_train_old = X_train_path
            y_train_old = y_train_path
            X_train_tmp = output_dir / 'X_train_bal.tmp.npy'
            y_train_tmp = output_dir / 'y_train_bal.tmp.npy'

            # Open old memmaps for reading
            y_old = np.memmap(y_train_old, dtype=np.int8, mode='r', shape=(train_cursor,))
            X_old = np.memmap(X_train_old, dtype=np.float32, mode='r', shape=(train_cursor, first_win_shape[0], first_win_shape[1]))

            idx_non = np.where(y_old == 0)[0]
            idx_sz = np.where(y_old == 1)[0]

            # Sample indices for each class
            def sample_indices(idxs, target, replace):
                if len(idxs) == 0:
                    return np.array([], dtype=int)
                return np.random.choice(idxs, size=target, replace=replace)

            replace_non = (len(idx_non) < target_train_per_class)
            replace_sz = (len(idx_sz) < target_train_per_class)

            sampled_non = sample_indices(idx_non, target_train_per_class, replace_non)
            sampled_sz = sample_indices(idx_sz, target_train_per_class, replace_sz)

            # Compose balanced ordering and write to temp memmaps
            total_target = 2 * target_train_per_class
            X_tmp = np.memmap(X_train_tmp, mode='w+', dtype=np.float32, shape=(total_target, first_win_shape[0], first_win_shape[1]))
            y_tmp = np.memmap(y_train_tmp, mode='w+', dtype=np.int8, shape=(total_target,))

            combined = np.concatenate([sampled_non, sampled_sz])
            np.random.shuffle(combined)

            for i, idx in enumerate(combined):
                X_tmp[i] = X_old[idx]
                y_tmp[i] = y_old[idx]

            # Flush and replace original files
            del X_tmp, y_tmp, X_old, y_old
            os.replace(str(X_train_tmp), str(X_train_old))
            os.replace(str(y_train_tmp), str(y_train_old))

            # Update counters to reflect new balanced counts
            train_written['non'] = target_train_per_class
            train_written['sz'] = target_train_per_class
            train_cursor = total_target

        # Optionally balance test set
        if balance_test and test_written['non'] > 0 and test_written['sz'] > 0:
            if balance_strategy == 'undersample':
                target_test_per_class = min(test_written['non'], test_written['sz'])
            elif balance_strategy == 'oversample':
                target_test_per_class = max(test_written['non'], test_written['sz'])
            else:
                target_test_per_class = min(test_written['non'], test_written['sz'])

            X_test_old = X_test_path
            y_test_old = y_test_path
            X_test_tmp = output_dir / 'X_test_bal.tmp.npy'
            y_test_tmp = output_dir / 'y_test_bal.tmp.npy'

            y_old = np.memmap(y_test_old, dtype=np.int8, mode='r', shape=(test_cursor,))
            X_old = np.memmap(X_test_old, dtype=np.float32, mode='r', shape=(test_cursor, first_win_shape[0], first_win_shape[1]))

            idx_non = np.where(y_old == 0)[0]
            idx_sz = np.where(y_old == 1)[0]

            replace_non = (len(idx_non) < target_test_per_class)
            replace_sz = (len(idx_sz) < target_test_per_class)

            sampled_non = np.random.choice(idx_non, size=target_test_per_class, replace=replace_non) if len(idx_non) > 0 else np.array([], dtype=int)
            sampled_sz = np.random.choice(idx_sz, size=target_test_per_class, replace=replace_sz) if len(idx_sz) > 0 else np.array([], dtype=int)

            total_test_target = 2 * target_test_per_class
            X_tmp = np.memmap(X_test_tmp, mode='w+', dtype=np.float32, shape=(total_test_target, first_win_shape[0], first_win_shape[1]))
            y_tmp = np.memmap(y_test_tmp, mode='w+', dtype=np.int8, shape=(total_test_target,))

            combined = np.concatenate([sampled_non, sampled_sz])
            np.random.shuffle(combined)

            for i, idx in enumerate(combined):
                X_tmp[i] = X_old[idx]
                y_tmp[i] = y_old[idx]

            del X_tmp, y_tmp, X_old, y_old
            os.replace(str(X_test_tmp), str(X_test_old))
            os.replace(str(y_test_tmp), str(y_test_old))

            test_written['non'] = target_test_per_class
            test_written['sz'] = target_test_per_class
            test_cursor = total_test_target

    # Manifest with compact subject-to-window mapping
    manifest = {
        "sample_rate": float(sample_rate),
        "window_sec": float(window_sec),
        "stride_sec": float(stride_sec),
        "n_channels": int(n_channels),
        "n_train": int(train_cursor),
        "n_test": int(test_cursor),
        "train_class_0": int(train_written["non"]),
        "train_class_1": int(train_written["sz"]),
        "test_class_0": int(test_written["non"]),
        "test_class_1": int(test_written["sz"]),
        # Balanced counts per-class (if balancing applied)
        "train_per_class": int(train_written["sz"]) if balance_strategy not in ("none", None) else None,
        "test_per_class": int(test_written["sz"]) if (balance_test and balance_strategy not in ("none", None)) else None,
        "n_subjects_train": int(len(train_subjects)),
        "n_subjects_test": int(len(test_subjects)),
        "X_train_shape": [train_cursor, first_win_shape[0], first_win_shape[1]],
        "X_test_shape": [test_cursor, first_win_shape[0], first_win_shape[1]],
        "X_train": str(X_train_path),
        "X_test": str(X_test_path),
        "y_train": str(y_train_path),
        "y_test": str(y_test_path),
        "subject_window_map_train": subject_window_map_train,
        "subject_window_map_test": subject_window_map_test,
        "split_method": "stratified_by_subject_seizure_rate" if use_stratified_split else "grouped_stratified_by_subject",
        "test_size": float(test_size),
        "random_seed": int(random_state),
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Log error summary
    error_report = error_tracker.report()
    print(error_report)
    logger.info(error_report)

    return output_dir / "manifest.json"


def main() -> None:
    """Main entry point for preprocessing pipeline."""
    # Get default dataset root path
    default_dataset_root = PATHS["project_root"] / DEFAULT_PARAMS["dataset_root"]
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root,
        help=f"Path to dataset (default: {default_dataset_root})",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=DEFAULT_PARAMS["window_sec"],
        help=f"Window length in seconds (default: {DEFAULT_PARAMS['window_sec']})",
    )
    parser.add_argument(
        "--stride-sec",
        type=float,
        default=DEFAULT_PARAMS["stride_sec"],
        help=f"Stride length in seconds (default: {DEFAULT_PARAMS['stride_sec']})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_PARAMS["test_size"],
        help=f"Test set fraction (default: {DEFAULT_PARAMS['test_size']})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_PARAMS["random_seed"],
        help=f"Random seed (default: {DEFAULT_PARAMS['random_seed']})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS["features"],
        help=f"Output directory for balanced split (default: {PATHS['features']})",
    )
    parser.add_argument(
        "--random-split",
        action="store_true",
        help="Use random subject split instead of stratified (may cause severe imbalance)",
    )
    parser.add_argument(
        "--balance",
        choices=["none", "undersample", "oversample"],
        default="none",
        help="Balance strategy for train set: undersample or oversample minority (default: none)",
    )
    parser.add_argument(
        "--balance-test",
        action="store_true",
        help="Also apply balancing to the test set (default: False)",
    )
    args = parser.parse_args()

    # Initialize logging
    logger = setup_logging(args.output_dir)

    print("=" * 80)
    print("SEIZURE CLASSIFICATION PREPROCESSING")
    print("=" * 80)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Window: {args.window_sec}s, Stride: {args.stride_sec}s")
    print(f"Test size: {args.test_size}, Seed: {args.seed}")
    print(f"Split method: {'Random' if args.random_split else 'Stratified (recommended)'}")
    print()
    
    logger.info("=" * 80)
    logger.info("SEIZURE CLASSIFICATION PREPROCESSING")
    logger.info("=" * 80)
    logger.info(f"Dataset root: {args.dataset_root}")
    logger.info(f"Window: {args.window_sec}s, Stride: {args.stride_sec}s")
    logger.info(f"Test size: {args.test_size}, Seed: {args.seed}")
    logger.info(f"Split method: {'Random' if args.random_split else 'Stratified'}")

    try:
        out = build_balanced_split(
            dataset_root=args.dataset_root,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            test_size=args.test_size,
            random_state=args.seed,
            output_dir=args.output_dir,
            use_stratified_split=not args.random_split,
            balance_strategy=args.balance,
            balance_test=args.balance_test,
            logger=logger,
        )
        split_type = "stratified" if not args.random_split else "random"
        print(f"✓ Saved {split_type} subject split to {out}")
        logger.info(f"✓ Saved {split_type} subject split to {out}")
    except Exception as e:
        logger.error(f"FATAL ERROR: {type(e).__name__}: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
