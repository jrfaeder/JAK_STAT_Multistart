"""
Test plotting different parameter sets sequentially.
"""

import numpy as np
import pandas as pd
from parameter_estimator import ParameterEstimator

# Load PETAb files
conditions_df = pd.read_csv('../petab_files/conditions.tsv', sep='\t')
measurements_df = pd.read_csv('../petab_files/measurements.tsv', sep='\t')
observables_df = pd.read_csv('../petab_files/observables.tsv', sep='\t')
parameters_df = pd.read_csv('../petab_files/parameters.tsv', sep='\t')

# Use subset of conditions
condition_subset = [(1.0, 0.0), (10.0, 0.0), (0.0, 1.0), (0.0, 10.0), (1.0, 1.0), (10.0, 10.0)]
mask = conditions_df.apply(
    lambda row: (row['L1_0'], row['L2_0']) in condition_subset,
    axis=1
)
conditions_subset = conditions_df[mask].reset_index(drop=True)
measurements_subset = measurements_df[
    measurements_df['simulationConditionId'].isin(conditions_subset['conditionId'])
].reset_index(drop=True)

# Create estimator
estimator = ParameterEstimator(
    bngl_file="../variable_JAK_STAT_SOCS_degrad_model.bngl",
    conditions_df=conditions_subset,
    measurements_df=measurements_subset,
    observables_df=observables_df,
    parameters_df=parameters_df
)

# Load parameter sets
param_sets_df = pd.read_csv('../param_sets.csv')

# Evaluate a few parameter sets and track their NLLH and simulation outputs
print("Evaluating parameter sets and tracking outputs:\n")

test_indices = [0, 100, 200, 300]
results_summary = []

for idx in test_indices:
    fitted_dict = param_sets_df.iloc[idx].to_dict()
    fitted_dict_subset = {}
    for param_name in estimator.estimable_params:
        if param_name in fitted_dict:
            fitted_dict_subset[param_name] = fitted_dict[param_name]
        else:
            fitted_dict_subset[param_name] = estimator.param_info[param_name]['nominal']

    param_vector = estimator.params_to_vector(fitted_dict_subset)

    # Compute NLLH
    nllh = estimator.compute_nllh(param_vector, verbose=False)

    # Get simulation results for first condition
    results = estimator.simulate_conditions(param_vector)
    cond_id = conditions_subset.iloc[0]['conditionId']

    # Get values at t=20 for first condition
    time_idx_20 = np.argmin(np.abs(results[cond_id]['time'] - 20.0))
    pS1_at_20 = results[cond_id]['total_pS1'][time_idx_20]
    pS3_at_20 = results[cond_id]['total_pS3'][time_idx_20]

    results_summary.append({
        'idx': idx,
        'nllh': nllh,
        'pS1_20': pS1_at_20,
        'pS3_20': pS3_at_20,
        'param_sample': fitted_dict_subset['il10_complex_jak1_binding']
    })

    print(f"Set {idx}: NLLH={nllh:.2f}, pS1@t=20={pS1_at_20:.3e}, pS3@t=20={pS3_at_20:.3e}")

print("\n" + "="*60)
print("Summary:")
print("="*60)

for i, r in enumerate(results_summary):
    print(f"\nParameter set {r['idx']}:")
    print(f"  NLLH: {r['nllh']:.4f}")
    print(f"  pS1 at t=20: {r['pS1_20']:.6e}")
    print(f"  pS3 at t=20: {r['pS3_20']:.6e}")
    print(f"  Sample param: {r['param_sample']:.4f}")

# Check if all are unique
pS1_values = [r['pS1_20'] for r in results_summary]
if len(set([f"{v:.12e}" for v in pS1_values])) == len(pS1_values):
    print("\n✓ All pS1 values are unique - parameter updates are working!")
else:
    print("\n✗ Some pS1 values are identical - parameter updates may not be working")
    for i in range(len(pS1_values)):
        for j in range(i+1, len(pS1_values)):
            if abs(pS1_values[i] - pS1_values[j]) < 1e-15:
                print(f"  Sets {test_indices[i]} and {test_indices[j]} have identical pS1 values")
