#python scripts/run_pipeline.py --test-size 0.4 --threshold-metric youden \
#  --balance oversample --max-non-seizure 2000 --skip-preprocessing --cv-folds 3
"""
Analyze channels across all subjects to find common channels.
"""

import re
from pathlib import Path
from collections import Counter
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.paths import PATHS

def parse_seizure_list(file_path: Path):
    """Parse channel information from a seizure list file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    channels = []
    # Match lines like "Channel 1: Fp1"
    pattern = r'Channel\s+\d+:\s+(\S+)'
    matches = re.findall(pattern, content)
    
    for ch in matches:
        ch_clean = ch.strip()
        # Skip EKG channels
        if 'EKG' not in ch_clean and ch_clean not in ['1', 'O1']:
            channels.append(ch_clean)
    
    return channels

def main():
    dataset_root = PATHS["project_root"] / "siena-scalp-eeg-database-1.0.0"
    
    # Find all seizure list files
    seizure_lists = sorted(dataset_root.glob("PN*/Seizures-list-*.txt"))
    
    print(f"Found {len(seizure_lists)} subjects\n")
    
    # Track channels per subject
    subject_channels = {}
    all_channels = []
    
    for seizure_file in seizure_lists:
        subject_id = seizure_file.parent.name
        channels = parse_seizure_list(seizure_file)
        subject_channels[subject_id] = set(channels)
        all_channels.extend(channels)
        print(f"{subject_id}: {len(channels)} channels - {', '.join(sorted(channels)[:10])}...")
    
    print("\n" + "="*80)
    print("CHANNEL ANALYSIS")
    print("="*80)
    
    # Count channel occurrences
    channel_counts = Counter(all_channels)
    n_subjects = len(seizure_lists)
    
    print(f"\nTotal subjects: {n_subjects}")
    print(f"\nChannels present in ALL subjects:")
    common_channels = []
    for channel, count in sorted(channel_counts.items(), key=lambda x: (-x[1], x[0])):
        if count == n_subjects:
            common_channels.append(channel)
            print(f"  {channel:8s} - {count}/{n_subjects} subjects")
    
    print(f"\nChannels present in MOST subjects (>= {n_subjects-2}):")
    almost_common = []
    for channel, count in sorted(channel_counts.items(), key=lambda x: (-x[1], x[0])):
        if count >= n_subjects - 2 and count < n_subjects:
            almost_common.append(channel)
            print(f"  {channel:8s} - {count}/{n_subjects} subjects")
    
    print(f"\n\nRECOMMENDED STANDARD CHANNEL SET ({len(common_channels)} channels):")
    print(f"{', '.join(sorted(common_channels))}")
    
    # Print subjects missing any common channels
    print(f"\n\nSubjects missing common channels:")
    for subject_id, channels in sorted(subject_channels.items()):
        missing = set(common_channels) - channels
        if missing:
            print(f"  {subject_id}: missing {', '.join(sorted(missing))}")
    
    # Generate a standard 10-20 channel list
    standard_1020 = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
                     'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                     'Fz', 'Cz', 'Pz']
    
    available_1020 = [ch for ch in standard_1020 if ch in common_channels]
    print(f"\n\nStandard 10-20 channels available in ALL subjects ({len(available_1020)}):")
    print(f"{', '.join(available_1020)}")

if __name__ == "__main__":
    main()
