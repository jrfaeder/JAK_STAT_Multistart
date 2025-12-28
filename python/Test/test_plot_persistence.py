"""
Test if plot_fit properly updates between different parameter sets.
"""

import numpy as np
import pandas as pd
from parameter_estimator import ParameterEstimator

# Load PETAb files
conditions_df = pd.read_csv('../petab_files/conditions.tsv', sep='\t')
measurements_df = pd.read_csv('../petab_files/measurements.tsv', sep='\t')
observables_df = pd.read_csv('../petab_files/observables.tsv', sep='\t')
parameters_df = pd.read_csv('../petab_files/parameters.tsv', sep='\t')

# Use subset of conditions (same as notebook)
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

# Test with the exact indices from your notebook: 60, 50, 96
test_indices = [60, 50, 96]

print("Testing simulate_conditions for indices 60, 50, 96:\n")
print("="*70)

for idx in test_indices:
    fitted_dict = param_sets_df.iloc[idx].to_dict()
    fitted_dict_subset = {}
    for param_name in estimator.estimable_params:
        if param_name in fitted_dict:
            fitted_dict_subset[param_name] = fitted_dict[param_name]
        else:
            fitted_dict_subset[param_name] = estimator.param_info[param_name]['nominal']

    param_vector = estimator.params_to_vector(fitted_dict_subset)

    # Simulate conditions (this is what plot_fit calls internally)
    results = estimator.simulate_conditions(param_vector)

    # Get first condition (IL6=1, IL10=0)
    cond_id = conditions_subset.iloc[0]['conditionId']

    # Get values at different time points
    time_0 = results[cond_id]['total_pS1'][0]
    time_idx_20 = np.argmin(np.abs(results[cond_id]['time'] - 20.0))
    time_20 = results[cond_id]['total_pS1'][time_idx_20]
    time_final = results[cond_id]['total_pS1'][-1]

    print(f"\nParameter set {idx}:")
    print(f"  Condition: {cond_id}")
    print(f"  total_pS1 at t=0:  {time_0:.6e}")
    print(f"  total_pS1 at t=20: {time_20:.6e}")
    print(f"  total_pS1 at t=90: {time_final:.6e}")
    print(f"  Sample param: {fitted_dict_subset['il10_complex_jak1_binding']:.4f}")

print("\n" + "="*70)
print("Checking if values are unique:")

# Reload and test again to see if there's persistence
print("\nSecond pass (testing for persistence issues):")
for idx in test_indices:
    fitted_dict = param_sets_df.iloc[idx].to_dict()
    fitted_dict_subset = {}
    for param_name in estimator.estimable_params:
        if param_name in fitted_dict:
            fitted_dict_subset[param_name] = fitted_dict[param_name]
        else:
            fitted_dict_subset[param_name] = estimator.param_info[param_name]['nominal']

    param_vector = estimator.params_to_vector(fitted_dict_subset)
    results = estimator.simulate_conditions(param_vector)
    cond_id = conditions_subset.iloc[0]['conditionId']
    time_idx_20 = np.argmin(np.abs(results[cond_id]['time'] - 20.0))
    time_20 = results[cond_id]['total_pS1'][time_idx_20]

    print(f"  Set {idx}: pS1@t=20 = {time_20:.6e}")
