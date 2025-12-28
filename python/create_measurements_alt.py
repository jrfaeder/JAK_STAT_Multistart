"""
Create alternative PETAb measurements file with individual experimental replicates.

This script extracts individual experimental data from the Excel files and creates
a measurements_alt.tsv file that preserves all individual data points.

Processing steps for each experiment (matches kmeans_clustering_trajs.ipynb):
1. Baseline correction: Subtract t=0 value for each condition to remove background fluorescence
2. Clip negative values to 0 (prevents negative phosphorylation values)
3. Normalization: Divide all values by the baseline-corrected t=20 value for IL6=10, IL10=0
   to make that reference point equal to 1.0 for both observables
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define paths
data_dir = Path('../Data')
petab_dir = Path('../petab_files')

# Load existing PETAb files to understand structure
conditions_df = pd.read_csv(petab_dir / 'conditions.tsv', sep='\t')

# Mapping from Excel condition names to (L1_0, L2_0) values
condition_name_map = {
    # IL10 alone
    'il10_01': (0.0, 0.1),
    'il10_1': (0.0, 1.0),
    'il10_10': (0.0, 10.0),
    # IL6 alone
    'il6_01': (0.1, 0.0),
    'il6_1': (1.0, 0.0),
    'il6_10': (10.0, 0.0),
    # Combined IL6 and IL10
    'il6_01_il10_01': (0.1, 0.1),
    'il6_01_il10_1': (0.1, 1.0),
    'il6_01_il10_10': (0.1, 10.0),
    'il6_1_il10_01': (1.0, 0.1),
    'il6_1_il10_1': (1.0, 1.0),
    'il6_1_il10_10': (1.0, 10.0),
    'il6_10_il10_01': (10.0, 0.1),
    'il6_10_il10_1': (10.0, 1.0),
    'il6_10_il10_10': (10.0, 10.0),
}

# Find conditionId for each (L1_0, L2_0) pair
def get_condition_id(L1, L2):
    """Get conditionId for given ligand concentrations."""
    match = conditions_df[(conditions_df['L1_0'] == L1) & (conditions_df['L2_0'] == L2)]
    if len(match) > 0:
        return match.iloc[0]['conditionId']
    return None

# Load Excel files
print("Loading Excel files...")
pS1_sheets = pd.read_excel(data_dir / 'pSTAT1_pooled_data.xlsx', sheet_name=None)
pS3_sheets = pd.read_excel(data_dir / 'pSTAT3_pooled_data.xlsx', sheet_name=None)

print(f"pSTAT1: {len(pS1_sheets)} experiments")
print(f"pSTAT3: {len(pS3_sheets)} experiments")

# Process each observable separately
measurements_alt = []

for observable_id, observable_name, sheets_dict in [
    ('obs_total_pS1', 'pSTAT1', pS1_sheets),
    ('obs_total_pS3', 'pSTAT3', pS3_sheets)
]:
    print(f"\n{'='*80}")
    print(f"Processing {observable_name}...")

    for sheet_name, df in sheets_dict.items():
        print(f"\n  Experiment: {sheet_name}")

        # Get time column
        time_col = df['Time'].values

        # Find t=0 index for baseline correction
        t0_idx = np.argmin(np.abs(time_col - 0.0))

        # Find normalization column and indices
        norm_col_name = 'il6_10'
        if norm_col_name not in df.columns:
            print(f"    Warning: No {norm_col_name} column in {sheet_name}, skipping normalization")
            continue

        # Get baseline (t=0) and normalization (t=20) values for il6_10
        t20_idx = np.argmin(np.abs(time_col - 20.0))
        norm_t0_value = df[norm_col_name].iloc[t0_idx]
        norm_t20_value = df[norm_col_name].iloc[t20_idx]

        # Baseline-corrected normalization value (t=20 minus t=0)
        norm_value_corrected = norm_t20_value - norm_t0_value

        if pd.isna(norm_value_corrected) or norm_value_corrected <= 0:
            print(f"    Warning: Invalid baseline-corrected normalization value ({norm_value_corrected}) in {sheet_name}")
            continue

        print(f"    Baseline (t={time_col[t0_idx]:.1f}): {norm_t0_value:.2f}")
        print(f"    Raw t=20 value: {norm_t20_value:.2f}")
        print(f"    Baseline-corrected normalization value: {norm_value_corrected:.2f}")

        # Process each condition column
        for col in df.columns:
            if col in ['Time', 'Date'] or col.startswith('Date:') or col.startswith('Unnamed'):
                continue

            # Map column name to (L1_0, L2_0)
            if col not in condition_name_map:
                print(f"    Warning: Unknown condition '{col}', skipping")
                continue

            L1, L2 = condition_name_map[col]
            cond_id = get_condition_id(L1, L2)

            if cond_id is None:
                print(f"    Warning: No conditionId for {col} (L1={L1}, L2={L2}), skipping")
                continue

            # Get baseline value (t=0) for this condition
            baseline_value = df[col].iloc[t0_idx]

            if pd.isna(baseline_value):
                print(f"    Warning: Missing t=0 value for {col}, skipping condition")
                continue

            # Extract measurements, baseline-correct, and normalize
            for i, (time, raw_value) in enumerate(zip(time_col, df[col])):
                if pd.isna(raw_value):
                    continue

                # Step 1: Baseline correction (subtract t=0)
                baseline_corrected = raw_value - baseline_value

                # Step 1b: Clip negative values to 0 (matches kmeans_clustering_trajs.ipynb processing)
                baseline_corrected = max(0.0, baseline_corrected)

                # Step 2: Normalize by baseline-corrected il6_10 at t=20
                normalized_value = baseline_corrected / norm_value_corrected

                # Add to measurements list
                measurements_alt.append({
                    'observableId': observable_id,
                    'simulationConditionId': cond_id,
                    'time': time,
                    'measurement': normalized_value,
                    'noiseParameters': f'sigma_{observable_name}',
                    'replicateId': sheet_name
                })

# Create DataFrame
measurements_alt_df = pd.DataFrame(measurements_alt)

# Sort by observable, condition, time, replicate
measurements_alt_df = measurements_alt_df.sort_values(
    by=['observableId', 'simulationConditionId', 'time', 'replicateId']
).reset_index(drop=True)

# Save to file
output_file = petab_dir / 'measurements_expts.tsv'
measurements_alt_df.to_csv(output_file, sep='\t', index=False)

print(f"\n{'='*80}")
print(f"Created {output_file}")
print(f"Total measurements: {len(measurements_alt_df)}")
print(f"\nBreakdown by observable:")
for obs_id in measurements_alt_df['observableId'].unique():
    count = len(measurements_alt_df[measurements_alt_df['observableId'] == obs_id])
    n_replicates = measurements_alt_df[measurements_alt_df['observableId'] == obs_id]['replicateId'].nunique()
    print(f"  {obs_id}: {count} measurements across {n_replicates} experiments")

print(f"\nSample of data:")
print(measurements_alt_df.head(20))

print(f"\nVerification of baseline correction and normalization:")
print("="*80)
cond_il6_10 = get_condition_id(10.0, 0.0)

print("\n1. Checking that all t=0 values are 0.0 (baseline correction):")
for obs_id, obs_name in [('obs_total_pS1', 'pSTAT1'), ('obs_total_pS3', 'pSTAT3')]:
    t0_data = measurements_alt_df[
        (measurements_alt_df['observableId'] == obs_id) &
        (measurements_alt_df['time'] == 0.0)
    ]
    print(f"\n  {obs_name} - max |t=0 value|: {t0_data['measurement'].abs().max():.6f}")
    if t0_data['measurement'].abs().max() > 1e-10:
        print(f"    WARNING: Some t=0 values are not zero!")
        for _, row in t0_data.iterrows():
            if abs(row['measurement']) > 1e-10:
                print(f"      {row['simulationConditionId']}, {row['replicateId']}: {row['measurement']:.6f}")

print("\n2. Checking that il6_10 at t=20 is normalized to 1.0 for each replicate:")
for obs_id, obs_name in [('obs_total_pS1', 'pSTAT1'), ('obs_total_pS3', 'pSTAT3')]:
    obs_data = measurements_alt_df[
        (measurements_alt_df['observableId'] == obs_id) &
        (measurements_alt_df['simulationConditionId'] == cond_il6_10) &
        (measurements_alt_df['time'] == 20.0)
    ]
    print(f"\n  {obs_name} ({cond_il6_10}) at t=20:")
    for _, row in obs_data.iterrows():
        print(f"    {row['replicateId']}: {row['measurement']:.6f}")

print("\n" + "="*80)
