"""
Helper script to identify unique parameter sets in param_sets.csv.
"""

import pandas as pd

param_sets_df = pd.read_csv('../param_sets.csv')
parameters_df = pd.read_csv('../petab_files/parameters.tsv', sep='\t')

# Get model parameter columns (exclude L1_0, L2_0, SOCS*_0)
model_params = set(parameters_df['parameterId'].values)
model_param_cols = [col for col in param_sets_df.columns if col in model_params]

# Find rows where new parameter sets begin
print("Finding unique parameter sets...")
last_params = None
unique_indices = []

for idx in range(len(param_sets_df)):
    current_params = tuple(param_sets_df[model_param_cols].iloc[idx].values)
    if last_params is None or current_params != last_params:
        unique_indices.append(idx)
        last_params = current_params

print(f"Found {len(unique_indices)} unique parameter sets")
print(f"\nFirst 100 unique parameter set indices:")
print(unique_indices[:100])

# Save to file for easy reference
with open('unique_param_indices.txt', 'w') as f:
    f.write("# Indices of rows with unique parameter sets\n")
    f.write("# Total: {} unique sets out of {} total rows\n".format(
        len(unique_indices), len(param_sets_df)))
    f.write("unique_indices = " + str(unique_indices))

print("\nSaved to unique_param_indices.txt")
