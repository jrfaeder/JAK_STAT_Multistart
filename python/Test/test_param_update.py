"""
Quick test to verify parameter updates are working correctly.
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
condition_subset = [(1.0, 0.0), (10.0, 0.0)]
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

# Test with 3 different parameter sets
# Note: rows 0-9 have identical model parameters, so use rows that actually differ
print("Testing parameter updates with 3 different parameter sets:\n")

for idx in [0, 10, 50]:
    fitted_dict = param_sets_df.iloc[idx].to_dict()
    fitted_dict_subset = {}
    for param_name in estimator.estimable_params:
        if param_name in fitted_dict:
            fitted_dict_subset[param_name] = fitted_dict[param_name]
        else:
            fitted_dict_subset[param_name] = estimator.param_info[param_name]['nominal']

    param_vector = estimator.params_to_vector(fitted_dict_subset)
    nllh = estimator.compute_nllh(param_vector, verbose=False)

    # Also get simulation results
    results = estimator.simulate_conditions(param_vector)
    cond_id = conditions_subset.iloc[0]['conditionId']
    final_pS1 = results[cond_id]['total_pS1'][-1]

    print(f"Parameter set {idx}:")
    print(f"  NLLH: {nllh:.4f}")
    print(f"  Final total_pS1 (IL6=1, IL10=0): {final_pS1:.6e}")
    print(f"  Sample param 'il10_complex_jak1_binding': {fitted_dict_subset['il10_complex_jak1_binding']:.4f}")
    print()

print("If parameter updates are working correctly, you should see:")
print("  - Different NLLH values")
print("  - Different final total_pS1 values")
print("  - Different parameter values")
