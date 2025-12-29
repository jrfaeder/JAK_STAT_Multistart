"""
Test script to verify NLLH computation using synthetic data.

This script:
1. Loads a parameter set from param_sets.csv
2. Generates synthetic measurements using those parameters
3. Computes NLLH for the same parameter set
4. Verifies that NLLH is close to the expected value for perfect fit
"""

import numpy as np
import pandas as pd
from parameter_estimator import ParameterEstimator

# Load PETAb files
print("Loading PETAb files...")
conditions_df = pd.read_csv('../petab_files/conditions.tsv', sep='\t')
measurements_df = pd.read_csv('../petab_files/measurements.tsv', sep='\t')
observables_df = pd.read_csv('../petab_files/observables.tsv', sep='\t')
parameters_df = pd.read_csv('../petab_files/parameters.tsv', sep='\t')

# Load parameter sets
param_sets_df = pd.read_csv('../param_sets.csv')
print(f"Loaded {len(param_sets_df)} parameter sets")

# Use the first parameter set
param_set_idx = 52
fitted_dict = param_sets_df.iloc[param_set_idx].to_dict()
print(f"\nUsing parameter set #{param_set_idx}")

# Create estimator
print("\nCreating parameter estimator...")
estimator = ParameterEstimator(
    bngl_file='../variable_JAK_STAT_SOCS_degrad_model.bngl',
    conditions_df=conditions_df,
    measurements_df=measurements_df,
    observables_df=observables_df,
    parameters_df=parameters_df
)

# Create parameter dict for estimator (only estimable params)
fitted_dict_subset = {}
for param_name in estimator.estimable_params:
    if param_name in fitted_dict:
        fitted_dict_subset[param_name] = fitted_dict[param_name]
    else:
        # Use nominal value for missing parameters
        fitted_dict_subset[param_name] = estimator.param_info[param_name]['nominal']

# Convert to parameter vector
param_vector = estimator.params_to_vector(fitted_dict_subset)

print("\nGenerating synthetic measurements from parameter set...")
# IMPORTANT: We need to use the SAME simulation approach as compute_nllh
# to ensure we get consistent scaling factors and model values

# Get the simulator and compute scaling factors
param_dict = estimator.vector_to_params(param_vector)
simulator = estimator._get_simulator(param_dict)
sf_pSTAT1, sf_pSTAT3 = estimator._compute_scaling_factors(
    param_dict, simulator, verbose=True
)

print(f"Scaling factors: sf_pSTAT1={sf_pSTAT1:.2f}, sf_pSTAT3={sf_pSTAT3:.2f}")

# Now simulate each condition using the SAME approach as compute_nllh
results = {}
for cond_id in conditions_df['conditionId']:
    cond_row = conditions_df[conditions_df['conditionId'] == cond_id].iloc[0]
    cond_params = {col: cond_row[col] for col in ['L1_0', 'L2_0'] if col in cond_row}

    # Get measurements for this condition to determine timepoints
    cond_measurements = measurements_df[
        measurements_df['simulationConditionId'] == cond_id
    ]
    meas_times = np.sort(cond_measurements['time'].unique())

    # Use GCD approach (same as compute_nllh)
    time_gcd = np.gcd.reduce(meas_times.astype(int))
    if time_gcd == 0:
        time_gcd = 1
    t_end = meas_times[-1]
    n_steps = int(t_end / time_gcd) + 1

    result = simulator.simulate(
        t_end=t_end,
        n_steps=n_steps,
        reference_values=cond_params
    )
    results[cond_id] = result

print(f"Simulated {len(results)} conditions")

# Create synthetic measurements dataframe
synthetic_measurements = []

for _, meas_row in measurements_df.iterrows():
    cond_id = meas_row['simulationConditionId']
    obs_id = meas_row['observableId']
    time = meas_row['time']

    # Get simulation result for this condition
    result = results[cond_id]

    # Find the timepoint (use interpolation for fine resolution)
    time_idx = np.argmin(np.abs(result['time'] - time))

    # Get the observable value
    if 'pS1' in obs_id:
        model_obs = 'total_pS1'
        sf = sf_pSTAT1
    else:
        model_obs = 'total_pS3'
        sf = sf_pSTAT3

    # Compute synthetic measurement (no noise for now)
    sim_value = result[model_obs][time_idx]
    synthetic_value = sf * sim_value

    # Create new measurement row
    new_row = meas_row.copy()
    new_row['measurement'] = synthetic_value
    synthetic_measurements.append(new_row)

# Create synthetic measurements dataframe
synthetic_measurements_df = pd.DataFrame(synthetic_measurements)

print(f"\nGenerated {len(synthetic_measurements_df)} synthetic measurements")
print("\nFirst few synthetic measurements:")
print(synthetic_measurements_df.head(10))

# Save synthetic measurements
synthetic_measurements_df.to_csv('synthetic_measurements.tsv', sep='\t', index=False)
print("\nSaved synthetic measurements to synthetic_measurements.tsv")

# Now create a new estimator with the synthetic data
print("\n" + "="*70)
print("Testing NLLH computation with synthetic data...")
print("="*70)

estimator_synthetic = ParameterEstimator(
    bngl_file='../variable_JAK_STAT_SOCS_degrad_model.bngl',
    conditions_df=conditions_df,
    measurements_df=synthetic_measurements_df,  # Use synthetic data
    observables_df=observables_df,
    parameters_df=parameters_df
)

# Compute NLLH for the same parameter set
nllh = estimator_synthetic.compute_nllh(param_vector, verbose=True)

print(f"\nComputed NLLH: {nllh:.4f}")

# Compute expected NLLH for perfect fit
# For Gaussian likelihood with noise model: sigma * (prediction + 0.01)
# NLLH = sum over measurements of: log(noise_std) + 0.5*log(2*pi)
# When residual = 0 (perfect fit), we only get the normalization term

expected_nllh = 0.0
n_datapoints = len(synthetic_measurements_df)

# Get noise parameter (sigma) - using default
sigma = 0.15

for _, meas_row in synthetic_measurements_df.iterrows():
    prediction = meas_row['measurement']
    noise_std = sigma * (prediction + 0.01)
    expected_nllh += np.log(noise_std) + 0.5 * np.log(2 * np.pi)

print(f"\nExpected NLLH for perfect fit: {expected_nllh:.4f}")
print(f"Difference: {abs(nllh - expected_nllh):.6f}")

if abs(nllh - expected_nllh) < 1e-6:
    print("\n✓ TEST PASSED: NLLH matches expected value for perfect fit!")
else:
    print("\n✗ TEST FAILED: NLLH does not match expected value")
    print(f"  This suggests there may still be timepoint mismatch issues")

# Additional verification: check that residuals are all close to zero
print("\n" + "="*70)
print("Verifying residuals...")
print("="*70)

# Re-simulate with the same n_steps as used in compute_nllh
param_dict = estimator_synthetic.vector_to_params(param_vector)
simulator = estimator_synthetic._get_simulator(param_dict)
sf_pSTAT1_test, sf_pSTAT3_test = estimator_synthetic._compute_scaling_factors(
    param_dict, simulator, verbose=False
)

max_residual = 0.0
for cond_id in conditions_df['conditionId']:
    cond_row = conditions_df[conditions_df['conditionId'] == cond_id].iloc[0]
    cond_params = {col: cond_row[col] for col in ['L1_0', 'L2_0'] if col in cond_row}

    # Get measurements for this condition
    cond_measurements = synthetic_measurements_df[
        synthetic_measurements_df['simulationConditionId'] == cond_id
    ]
    meas_times = np.sort(cond_measurements['time'].unique())

    # Simulate with same settings as compute_nllh
    time_gcd = np.gcd.reduce(meas_times.astype(int))
    if time_gcd == 0:
        time_gcd = 1
    t_end = meas_times[-1]
    n_steps = int(t_end / time_gcd) + 1

    result = simulator.simulate(
        t_end=t_end,
        n_steps=n_steps,
        reference_values=cond_params
    )

    for _, meas_row in cond_measurements.iterrows():
        obs_id = meas_row['observableId']
        time = meas_row['time']
        measurement = meas_row['measurement']

        time_idx = np.argmin(np.abs(result['time'] - time))

        if 'pS1' in obs_id:
            model_obs = 'total_pS1'
            sf = sf_pSTAT1_test
        else:
            model_obs = 'total_pS3'
            sf = sf_pSTAT3_test

        sim_value = result[model_obs][time_idx]
        prediction = sf * sim_value
        residual = abs(measurement - prediction)
        max_residual = max(max_residual, residual)

print(f"Maximum residual: {max_residual:.2e}")

if max_residual < 1e-6:
    print("✓ All residuals are effectively zero - perfect fit confirmed!")
else:
    print(f"✗ WARNING: Non-zero residuals detected (max={max_residual:.2e})")
    print("  This suggests timepoint mismatch or numerical issues")

print("\n" + "="*70)
print("Test complete!")
print("="*70)