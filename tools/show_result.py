import pickle
from collections import defaultdict
from scipy.spatial.distance import jensenshannon
import numpy as np
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='workdir/downsampling_test/naive/out.pkl')
parser.add_argument('--name', type=str, default='unknown', help='Name/identifier for this analysis (e.g., cutoff value)')
parser.add_argument('--output_csv', type=str, default=None, help='Path to save results as CSV')
parser.add_argument('--type', type=str, default='naive', choices=['naive', 'standard_aa', 'multistage_aa'], help='Type of downsampling')
args = parser.parse_args()


def classify_dihedral(key):
    """
    Classify a key as:
      - 'TICA' if it starts with TICA (e.g., "TICA-0", "TICA-0,1"),
      - 'main' if it only involves main-chain dihedrals (PHI/PSI),
      - 'side' if it only involves side-chain dihedrals (CHI),
      - 'mixed' if it involves both main and side chain.

    If any component is unrecognized, it returns 'unknown'.
    """
    # First, check if the key is for a TICA component.
    if key.startswith("TICA-0,1"):
        return "TICA-0,1"
    if key.startswith("TICA-0"):
        return "TICA-0"

    # Otherwise, assume it's a dihedral-based key.
    dihedrals = [d.strip() for d in key.split('|')]
    classifications = []

    for d in dihedrals:
        if d.startswith("PHI") or d.startswith("PSI"):
            classifications.append("bb")
        elif d.startswith("CHI"):
            classifications.append("sc")
        else:
            classifications.append("unknown")

    if "unknown" in classifications:
        return "unknown"
    if len(set(classifications)) == 1:
        return classifications[0]
    return "mixed"


# Load the pickle file.
with open(args.path, 'rb') as f:
    data = pickle.load(f)

# Dictionaries to accumulate sums and counts for each group.
group_sums = defaultdict(float)
group_counts = defaultdict(int)
meta_jsd = []

for peptide, metrics in data.items():
    if 'traj_metastable_probs' not in metrics:
        print(f"Missing 'traj_metastable_probs' for peptide: {peptide}")
# TODO: find out why peptide 'FKKL' doesn't have MSM-related results

for peptide, metrics in data.items():
    # if peptide not in ['ESSS', 'SPVN', 'HTIQ', 'HENV', 'FKKL']:
    if 'traj_metastable_probs' in metrics:
        traj_meta = metrics['traj_metastable_probs']
        ref_meta = metrics['ref_metastable_probs']
    jsd = metrics.get('JSD', {})
    meta_jsd.append(jensenshannon(ref_meta, traj_meta))

    for key, value in jsd.items():
        classification = classify_dihedral(key)
        if classification != "unknown":
            group_sums[classification] += value
            group_counts[classification] += 1
        if classification == "bb" or classification == "sc":
            group_sums["all"] += value
            group_counts["all"] += 1

# Calculate the mean value for each group.
mean_group = {group: group_sums[group] / group_counts[group]
              for group in group_sums if group_counts[group] > 0}

print("Mean JSD for each dihedral group and ITCA:")
for group, mean_val in mean_group.items():
    print(f"{group}: {mean_val:.4f}")
print("Mean JSD for metastable states:")
mean_jsd = np.mean(meta_jsd)
print(f"MSM states: {mean_jsd:.4f}")

# Create results dictionary for CSV output
results = {
    'ds_cutoff': args.name,
    'analysis_type': 'naive' if 'naive' in args.path else args.type,
    'msm_states': mean_jsd
}

# Add all group means to results
for group, mean_val in mean_group.items():
    results[f'jsd_{group}'] = mean_val

# Create DataFrame and save to CSV
df = pd.DataFrame([results])
from beautifultable import BeautifulTable
table = BeautifulTable()
table.columns.header = ['ds_cutoff', 'analysis_type', 'msm_states'] + list(mean_group.keys())
table.rows.append([args.name, args.type, mean_jsd] + list(mean_group.values()))
print(table)

if args.output_csv:
    # If output file exists, append to it
    if os.path.exists(args.output_csv):
        existing_df = pd.read_csv(args.output_csv)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    # Save the updated DataFrame
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to: {args.output_csv}")
else:
    # If no output file specified, save to default location
    output_file = os.path.join(os.path.dirname(args.path), 'analysis_results.csv')
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
