import pandas as pd
import numpy as np

# Load files
param_sets_df = pd.read_csv('../param_sets.csv')
parameters_df = pd.read_csv('../petab_files/parameters.tsv', sep='\t')

# Get first parameter set
first_row = param_sets_df.iloc[0]

# Get estimable params
estimable_params = parameters_df[parameters_df['estimate'] == 1]['parameterId'].tolist()

print("Estimable parameters:", len(estimable_params))
print("\nChecking for issues:")

# Build param dict as the notebook does
param_info = {}
for _, row in parameters_df.iterrows():
    param_info[row['parameterId']] = {
        'scale': row['parameterScale'],
        'lower': row['lowerBound'],
        'upper': row['upperBound'],
        'nominal': row['nominalValue']
    }

fitted_dict_subset = {}
for param_name in estimable_params:
    if param_name in ['L1_0', 'L2_0']:
        fitted_dict_subset[param_name] = param_info[param_name]['nominal']
        print(f"Skipping {param_name} (using nominal: {fitted_dict_subset[param_name]})")
    elif param_name in first_row:
        fitted_dict_subset[param_name] = first_row[param_name]
    else:
        fitted_dict_subset[param_name] = param_info[param_name]['nominal']
        if param_name.startswith('sf_'):
            print(f"{param_name} not in CSV, using nominal: {fitted_dict_subset[param_name]}")

# Now convert to vector as notebook does
print("\nConverting to vector (checking for issues)...")
for i, param_name in enumerate(estimable_params[:10]):
    value = fitted_dict_subset[param_name]
    scale = param_info[param_name]['scale']

    if scale == 'log10':
        vec_value = np.log10(value)
    else:
        vec_value = value

    print(f"{param_name:30s}: natural={value:12.6g}, log10={vec_value:12.6f}")

    # Check for problems
    if value <= 0 and scale == 'log10':
        print(f"  *** ERROR: Cannot take log10 of {value} ***")
    if np.isnan(vec_value) or np.isinf(vec_value):
        print(f"  *** ERROR: Invalid value {vec_value} ***")