import pandas as pd
import numpy as np
from bngl_simulator import BNGLSimulator

# Load files
param_sets_df = pd.read_csv('../param_sets.csv')
parameters_df = pd.read_csv('../petab_files/parameters.tsv', sep='\t')

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

print(f"Setting {len(fitted_params)} parameters")
print("\nSample fitted parameters:")
for i, (k, v) in enumerate(list(fitted_params.items())[:5]):
    print(f"  {k}: {v}")

# Try to create simulator
print("\nCreating simulator with fitted parameters...")
try:
    sim = BNGLSimulator("../variable_JAK_STAT_SOCS_degrad_model.bngl", param_values=fitted_params)
    print("Simulator created successfully")

    # Try a simple simulation
    print("\nTrying simulation with L1_0=10, L2_0=0...")
    sim.set_parameter('L1_0', 10.0)
    sim.set_parameter('L2_0', 0.0)

    result = sim.simulate(t_end=90, n_steps=7)

    print(f"Simulation completed, shape: {result['time'].shape}")
    print(f"Time points: {result['time']}")
    print(f"total_pS1 at end: {result['total_pS1'][-1]}")
    print(f"total_pS3 at end: {result['total_pS3'][-1]}")

    # Check for NaN or Inf
    if np.any(np.isnan(result['total_pS1'])) or np.any(np.isinf(result['total_pS1'])):
        print("WARNING: total_pS1 has NaN or Inf values!")
    if np.any(np.isnan(result['total_pS3'])) or np.any(np.isinf(result['total_pS3'])):
        print("WARNING: total_pS3 has NaN or Inf values!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()