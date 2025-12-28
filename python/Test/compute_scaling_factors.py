import pandas as pd
import numpy as np
from bngl_simulator import BNGLSimulator

# Load files
param_sets_df = pd.read_csv('../param_sets.csv')
parameters_df = pd.read_csv('../petab_files/parameters.tsv', sep='\t')
measurements_df = pd.read_csv('../petab_files/measurements.tsv', sep='\t')

# Get first parameter set
first_row = param_sets_df.iloc[0]

# Get estimable params (excluding scaling factors since they're not in model)
estimable_params = parameters_df[parameters_df['estimate'] == 1]['parameterId'].tolist()

# Build fitted dict - exclude sf_ and L1_0/L2_0
fitted_params = {}
for param_name in estimable_params:
    if param_name.startswith('sf_') or param_name in ['L1_0', 'L2_0']:
        continue
    if param_name in first_row:
        fitted_params[param_name] = first_row[param_name]

print("Computing scaling factors based on normalization condition:")
print("  - Time point: t=20")
print("  - Condition: L1_0=1.0, L2_0=0.0")
print("  - Method: Scale to match experimental data at this point\n")

# Create simulator with fitted parameters
sim = BNGLSimulator("../variable_JAK_STAT_SOCS_degrad_model.bngl", param_values=fitted_params)

# Set normalization condition
sim.set_parameter('L1_0', 1.0)
sim.set_parameter('L2_0', 0.0)

# Run simulation to t=20
result = sim.simulate(t_end=20, n_steps=20)

# Get values at t=20
pS1_model_at_20 = result['total_pS1'][-1]
pS3_model_at_20 = result['total_pS3'][-1]

print(f"Model predictions at t=20 (L1_0=1, L2_0=0):")
print(f"  total_pS1: {pS1_model_at_20:.6e}")
print(f"  total_pS3: {pS3_model_at_20:.6e}")

# Get experimental data at t=20 for condition cond_il6_1 (L1_0=1, L2_0=0)
exp_data = measurements_df[measurements_df['simulationConditionId'] == 'cond_il6_1']
exp_data_t20 = exp_data[exp_data['time'] == 20.0]

pS1_exp_at_20 = exp_data_t20[exp_data_t20['observableId'] == 'obs_total_pS1']['measurement'].values
pS3_exp_at_20 = exp_data_t20[exp_data_t20['observableId'] == 'obs_total_pS3']['measurement'].values

if len(pS1_exp_at_20) > 0 and len(pS3_exp_at_20) > 0:
    pS1_exp_at_20 = pS1_exp_at_20[0]
    pS3_exp_at_20 = pS3_exp_at_20[0]

    print(f"\nExperimental data at t=20 (L1_0=1, L2_0=0):")
    print(f"  obs_total_pS1: {pS1_exp_at_20:.6f}")
    print(f"  obs_total_pS3: {pS3_exp_at_20:.6f}")

    # Compute scaling factors
    sf_pSTAT1 = pS1_exp_at_20 / pS1_model_at_20
    sf_pSTAT3 = pS3_exp_at_20 / pS3_model_at_20

    print(f"\nComputed scaling factors:")
    print(f"  sf_pSTAT1 = {pS1_exp_at_20:.6f} / {pS1_model_at_20:.6e} = {sf_pSTAT1:.6f}")
    print(f"  sf_pSTAT3 = {pS3_exp_at_20:.6f} / {pS3_model_at_20:.6e} = {sf_pSTAT3:.6f}")
else:
    print("\nERROR: Could not find experimental data at t=20 for cond_il6_1")