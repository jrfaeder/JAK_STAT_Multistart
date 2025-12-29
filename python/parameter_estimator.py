"""
Parameter estimation for BNGL models using PETAb-formatted data.

This module provides the ParameterEstimator class which computes goodness-of-fit
(negative log-likelihood) for BNGL models given experimental data in PETAb format.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
from bngl_simulator import BNGLSimulator


class ParameterEstimator:
    """
    Parameter estimation for BNGL models using PETAb-formatted data.

    This class computes goodness-of-fit (negative log-likelihood) for a given
    set of parameter values by:
    1. Running simulations for each experimental condition
    2. Applying observable transformations (scaling)
    3. Computing likelihood using noise model
    """

    def __init__(self,
                 bngl_file: str,
                 conditions_df: pd.DataFrame,
                 measurements_df: pd.DataFrame,
                 observables_df: pd.DataFrame,
                 parameters_df: pd.DataFrame,
                 estimable_params: Optional[List[str]] = None):
        """
        Initialize parameter estimator.

        Args:
            bngl_file: Path to BNGL model file
            conditions_df: PETAb conditions table
            measurements_df: PETAb measurements table
            observables_df: PETAb observables table
            parameters_df: PETAb parameters table
            estimable_params: List of parameter names to estimate (subset)
                             If None, uses all parameters with estimate=1
        """
        self.bngl_file = bngl_file
        self.conditions_df = conditions_df
        self.measurements_df = measurements_df
        self.observables_df = observables_df
        self.parameters_df = parameters_df

        # Determine which parameters to estimate
        if estimable_params is None:
            # Use all parameters marked for estimation
            self.estimable_params = parameters_df[
                parameters_df['estimate'] == 1
            ]['parameterId'].tolist()
        else:
            self.estimable_params = estimable_params

        # Create parameter info dict for bounds and scales
        self.param_info = {}
        for _, row in parameters_df.iterrows():
            self.param_info[row['parameterId']] = {
                'scale': row['parameterScale'],
                'lower': row['lowerBound'],
                'upper': row['upperBound'],
                'nominal': row['nominalValue']
            }

        # Extract unique timepoints for simulation
        self.timepoints = np.sort(measurements_df['time'].unique())

        # Initialize simulator (will be created with default parameters)
        self.simulator = None

        print(f"Initialized estimator with {len(self.estimable_params)} estimable parameters")
        print(f"Timepoints: {self.timepoints}")

    def _get_simulator(self, param_dict: Dict[str, float]) -> BNGLSimulator:
        """
        Create or update simulator with given parameter values.

        Args:
            param_dict: Dictionary of parameter values (may include non-model params)

        Returns:
            BNGLSimulator instance
        """
        # Filter out non-model parameters (scaling factors, noise parameters)
        # These are PETAb-specific and not part of the BNGL model
        model_params = {
            k: v for k, v in param_dict.items()
            if not k.startswith('sf_') and not k.startswith('sigma_')
        }

        # Create simulator if it doesn't exist
        if self.simulator is None:
            self.simulator = BNGLSimulator(self.bngl_file, param_values=model_params)
            self._last_params = model_params.copy()
        else:
            # Update only the parameters that have changed
            # This is much more efficient than recreating the entire simulator
            if not hasattr(self, '_last_params'):
                # Safety check - should not happen, but initialize if needed
                self._last_params = {}

            params_changed = False
            for param_name, value in model_params.items():
                # Set parameter if it's new or if the value has changed
                if param_name not in self._last_params or self._last_params[param_name] != value:
                    self.simulator.set_parameter(param_name, value)
                    params_changed = True

            # Reset simulator after parameter changes to ensure dependencies are updated
            # This is critical per user feedback: "reset() has to be called after a parameter change"
            if params_changed:
                self.simulator.sim.reset()

            # Update cache
            self._last_params = model_params.copy()

        return self.simulator

    def vector_to_params(self, param_vector: np.ndarray) -> Dict[str, float]:
        """
        Convert parameter vector to dictionary.

        Handles log10 scaling as specified in PETAb.

        Args:
            param_vector: Array of parameter values (in estimation space)

        Returns:
            Dictionary mapping parameter names to values (in natural space)
        """
        param_dict = {}
        for i, param_name in enumerate(self.estimable_params):
            value = param_vector[i]
            scale = self.param_info[param_name]['scale']

            # Transform from estimation space to natural space
            if scale == 'log10':
                param_dict[param_name] = 10 ** value
            else:
                param_dict[param_name] = value

        return param_dict

    def params_to_vector(self, param_dict: Dict[str, float]) -> np.ndarray:
        """
        Convert parameter dictionary to vector.

        Handles log10 scaling as specified in PETAb.

        Args:
            param_dict: Dictionary of parameter values (in natural space)

        Returns:
            Array of parameter values (in estimation space)
        """
        param_vector = np.zeros(len(self.estimable_params))
        for i, param_name in enumerate(self.estimable_params):
            value = param_dict[param_name]
            scale = self.param_info[param_name]['scale']

            # Transform from natural space to estimation space
            if scale == 'log10':
                param_vector[i] = np.log10(value)
            else:
                param_vector[i] = value

        return param_vector

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get parameter bounds in estimation space.

        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays
        """
        lower = np.zeros(len(self.estimable_params))
        upper = np.zeros(len(self.estimable_params))

        for i, param_name in enumerate(self.estimable_params):
            info = self.param_info[param_name]

            if info['scale'] == 'log10':
                lower[i] = np.log10(info['lower'])
                upper[i] = np.log10(info['upper'])
            else:
                lower[i] = info['lower']
                upper[i] = info['upper']

        return lower, upper

    def compute_nllh(self, param_vector: np.ndarray, verbose: bool = False) -> float:
        """
        Compute negative log-likelihood for given parameters.

        This is the objective function to minimize.

        Scaling factors are automatically computed from the normalization condition:
        - Simulate condition with L1_0=10, L2_0=0 at t=20
        - Set sf_pSTAT1 and sf_pSTAT3 to normalize observables to experimental values

        Args:
            param_vector: Array of parameter values (in estimation space)
            verbose: Print debug information

        Returns:
            Negative log-likelihood value
        """
        # Convert to parameter dict
        param_dict = self.vector_to_params(param_vector)

        # Get simulator with these parameters
        simulator = self._get_simulator(param_dict)

        # Compute scaling factors from normalization condition
        sf_pSTAT1, sf_pSTAT3 = self._compute_scaling_factors(param_dict, simulator, verbose)

        # Check if scaling factor computation failed
        if np.isinf(sf_pSTAT1) or np.isinf(sf_pSTAT3):
            return np.inf

        # Compute log-likelihood contributions
        nllh = 0.0
        n_datapoints = 0

        # Group measurements by condition
        for cond_id in self.conditions_df['conditionId']:
            # Get condition parameters (L1_0, L2_0)
            cond_row = self.conditions_df[
                self.conditions_df['conditionId'] == cond_id
            ].iloc[0]

            # Set condition-specific parameters
            cond_params = {}
            for col in ['L1_0', 'L2_0']:
                if col in cond_row:
                    cond_params[col] = cond_row[col]

            # Get measurements for this condition
            cond_measurements = self.measurements_df[
                self.measurements_df['simulationConditionId'] == cond_id
            ]

            # Get unique measurement times for this condition
            meas_times = np.sort(cond_measurements['time'].unique())

            # Find the greatest common divisor of all measurement times to determine step size
            # All times should be multiples of this value
            time_gcd = np.gcd.reduce(meas_times.astype(int))
            if time_gcd == 0:
                time_gcd = 1

            # Calculate n_steps to ensure all measurement times are hit exactly
            # This creates evenly-spaced timepoints with spacing = time_gcd
            t_end = meas_times[-1]
            n_steps = int(t_end / time_gcd) + 1

            # Run simulation with fine enough resolution to hit all measurement timepoints
            try:
                #print(f"Simulating condition {cond_id} with t_end={t_end}, n_steps={n_steps}")
                result = simulator.simulate(
                    t_end=t_end,
                    n_steps=n_steps,
                    reference_values=cond_params
                )
            except Exception as e:
                if verbose:
                    print(f"Simulation failed for condition {cond_id}: {e}")
                return np.inf

            # Compute likelihood for each measurement
            for _, meas_row in cond_measurements.iterrows():
                obs_id = meas_row['observableId']
                time = meas_row['time']
                measurement = meas_row['measurement']

                # Find the timepoint index (should be exact match now with GCD-based stepping)
                time_idx = np.argmin(np.abs(result['time'] - time))
                if not np.isclose(result['time'][time_idx], time):
                    if verbose:
                        print(f"Warning: Timepoint {time} doesn't match closest ({result['time'][time_idx]}) in simulation results for condition {cond_id}")
                    return np.inf # Penalize if timepoint not found

                # Get observable formula and noise
                obs_row = self.observables_df[
                    self.observables_df['observableId'] == obs_id
                ].iloc[0]

                # Extract model observable name (e.g., 'total_pS1' from 'sf_pSTAT1 * total_pS1')
                # For now, assume simple format: sf * observable
                if 'pS1' in obs_id:
                    model_obs = 'total_pS1'
                    sf = sf_pSTAT1
                else:
                    model_obs = 'total_pS3'
                    sf = sf_pSTAT3

                # Compute predicted observable
                sim_value = result[model_obs][time_idx]
                prediction = sf * sim_value

                # Get noise parameter (sigma)
                noise_param = meas_row['noiseParameters']
                sigma = param_dict.get(noise_param, 0.15)  # Default from PETAb

                # Compute noise model: sigma * (sf * total_pS + 0.01)
                noise_std = sigma * (prediction + 0.01)

                # Gaussian log-likelihood: -0.5 * ((y - y_pred) / sigma)^2 - log(sigma) - 0.5*log(2*pi)
                residual = measurement - prediction
                nllh += 0.5 * (residual / noise_std) ** 2
                nllh += np.log(noise_std)
                nllh += 0.5 * np.log(2 * np.pi)

                n_datapoints += 1

        if verbose:
            print(f"NLLH = {nllh:.4f} ({n_datapoints} datapoints)")

        return nllh

    def _compute_scaling_factors(self, param_dict: Dict[str, float],
                                 simulator: BNGLSimulator,
                                 verbose: bool = False) -> Tuple[float, float]:
        """
        Compute scaling factors from normalization condition.

        Scaling factors are computed from the normalization condition:
        - Simulate condition with L1_0=10, L2_0=0 at t=20
        - Set sf_pSTAT1 and sf_pSTAT3 to normalize observables to experimental values

        Args:
            param_dict: Dictionary of parameter values
            simulator: BNGLSimulator instance
            verbose: Print debug information

        Returns:
            Tuple of (sf_pSTAT1, sf_pSTAT3)
        """
        # Find the normalization condition (L1_0=10, L2_0=0)
        norm_cond = self.conditions_df[
            (self.conditions_df['L1_0'] == 10.0) &
            (self.conditions_df['L2_0'] == 0.0)
        ]

        if len(norm_cond) > 0:
            norm_cond_id = norm_cond.iloc[0]['conditionId']

            # Run simulation at normalization condition
            # Get measurement times for normalization condition to ensure we hit t=20 exactly
            norm_meas = self.measurements_df[
                self.measurements_df['simulationConditionId'] == norm_cond_id
            ]
            norm_times = np.sort(norm_meas['time'].unique())

            # Find the last timepoint <= 20 (should be exactly 20)
            max_time = norm_times[norm_times <= 20.0][-1] if len(norm_times[norm_times <= 20.0]) > 0 else 20.0

            # Use GCD approach to ensure we hit t=20 exactly
            norm_times_subset = norm_times[norm_times <= max_time]
            time_gcd = np.gcd.reduce(norm_times_subset.astype(int))
            if time_gcd == 0:
                time_gcd = 1
            n_steps_norm = int(max_time / time_gcd)

            try:
                norm_result = simulator.simulate(
                    t_end=max_time,
                    n_steps=n_steps_norm,
                    reference_values={'L1_0': 10.0, 'L2_0': 0.0}
                )

                # Get model values at t=20 (should be exact match now)
                time_idx_20 = np.argmin(np.abs(norm_result['time'] - 20.0))
                pS1_model_20 = norm_result['total_pS1'][time_idx_20]
                pS3_model_20 = norm_result['total_pS3'][time_idx_20]

                # Get experimental values at t=20 for normalization condition
                norm_meas_20 = self.measurements_df[
                    (self.measurements_df['simulationConditionId'] == norm_cond_id) &
                    (self.measurements_df['time'] == 20.0)
                ]

                pS1_exp_20 = norm_meas_20[
                    norm_meas_20['observableId'] == 'obs_total_pS1'
                ]['measurement'].values[0]
                pS3_exp_20 = norm_meas_20[
                    norm_meas_20['observableId'] == 'obs_total_pS3'
                ]['measurement'].values[0]

                # Compute scaling factors
                if pS1_model_20 > 1e-12 and pS3_model_20 > 1e-12:
                    sf_pSTAT1 = pS1_exp_20 / pS1_model_20
                    sf_pSTAT3 = pS3_exp_20 / pS3_model_20
                else:
                    if verbose:
                        print(f"Warning: Model returned near-zero values at normalization condition")
                    return np.inf, np.inf

                if verbose:
                    print(f"Computed scaling factors: sf_pSTAT1={sf_pSTAT1:.2f}, sf_pSTAT3={sf_pSTAT3:.2f}")

                return sf_pSTAT1, sf_pSTAT3

            except Exception as e:
                if verbose:
                    print(f"Failed to compute scaling factors: {e}")
                return np.inf, np.inf
        else:
            # No normalization condition found, use default scaling
            if verbose:
                print("Warning: No normalization condition (L1_0=10, L2_0=0) found")
            sf_pSTAT1 = param_dict.get('sf_pSTAT1', 1.0)
            sf_pSTAT3 = param_dict.get('sf_pSTAT3', 1.0)
            return sf_pSTAT1, sf_pSTAT3

    def simulate_conditions(self, param_vector: np.ndarray,
                          compute_scaling: bool = True,
                          n_steps: int = 200) -> Tuple[Dict[str, dict], Tuple[float, float]]:
        """
        Simulate all experimental conditions with given parameters.

        Args:
            param_vector: Array of parameter values (in estimation space)
            compute_scaling: If True, compute and return scaling factors from normalization
            n_steps: Number of time steps for simulation (higher = smoother curves)

        Returns:
            Tuple of (results_dict, scaling_factors) where:
            - results_dict: Dictionary mapping condition IDs to simulation results
            - scaling_factors: Tuple of (sf_pSTAT1, sf_pSTAT3)
        """
        param_dict = self.vector_to_params(param_vector)
        simulator = self._get_simulator(param_dict)

        # Compute scaling factors if requested
        if compute_scaling:
            sf_pSTAT1, sf_pSTAT3 = self._compute_scaling_factors(param_dict, simulator)
        else:
            sf_pSTAT1 = param_dict.get('sf_pSTAT1', 1.0)
            sf_pSTAT3 = param_dict.get('sf_pSTAT3', 1.0)

        results = {}

        for cond_id in self.conditions_df['conditionId']:
            cond_row = self.conditions_df[
                self.conditions_df['conditionId'] == cond_id
            ].iloc[0]

            cond_params = {}
            for col in ['L1_0', 'L2_0']:
                if col in cond_row:
                    cond_params[col] = cond_row[col]

            result = simulator.simulate(
                t_end=self.timepoints[-1],
                n_steps=n_steps,
                reference_values=cond_params
            )

            results[cond_id] = result

        return results, (sf_pSTAT1, sf_pSTAT3)

    def plot_fit(self, param_vector: np.ndarray, figsize: Optional[Tuple[int, int]] = None):
        """
        Plot model fit vs experimental data.

        Args:
            param_vector: Array of parameter values (in estimation space)
            figsize: Figure size (optional, auto-calculated if not provided)
        """
        # Simulate all conditions and get scaling factors
        results, (sf_pSTAT1, sf_pSTAT3) = self.simulate_conditions(param_vector, compute_scaling=True)

        # Determine grid layout based on number of conditions
        n_conditions = len(self.conditions_df)
        print('Plotting fit for', n_conditions, 'conditions')

        # Calculate optimal grid layout (prefer wider than tall)
        if n_conditions <= 3:
            nrows, ncols = 1, n_conditions
        elif n_conditions <= 6:
            nrows, ncols = 2, 3
        elif n_conditions <= 9:
            nrows, ncols = 3, 3
        elif n_conditions <= 12:
            nrows, ncols = 3, 4
        else:
            # For larger numbers, use 3 columns
            ncols = 3
            nrows = (n_conditions + ncols - 1) // ncols  # Ceiling division

        # Auto-calculate figsize if not provided
        if figsize is None:
            figsize = (5 * ncols, 4 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        # Handle single subplot case
        if n_conditions == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, cond_id in enumerate(self.conditions_df['conditionId']):
            ax = axes[idx]
            result = results[cond_id]

            # Plot simulations
            ax.plot(result['time'], sf_pSTAT1 * result['total_pS1'],
                   '-', color='C0', label='pSTAT1 (model)')
            ax.plot(result['time'], sf_pSTAT3 * result['total_pS3'],
                   '-', color='C1', label='pSTAT3 (model)')

            # Plot measurements
            cond_data = self.measurements_df[
                self.measurements_df['simulationConditionId'] == cond_id
            ]

            for obs_id, color in [('obs_total_pS1', 'C0'), ('obs_total_pS3', 'C1')]:
                obs_data = cond_data[cond_data['observableId'] == obs_id]
                if len(obs_data) > 0:
                    label = 'pSTAT1 (data)' if 'pS1' in obs_id else 'pSTAT3 (data)'
                    ax.plot(obs_data['time'], obs_data['measurement'],
                           'o', color=color, label=label, markersize=6)

            # Get condition info
            cond_row = self.conditions_df[
                self.conditions_df['conditionId'] == cond_id
            ].iloc[0]
            ax.set_title(f"{cond_id}\nIL6={cond_row['L1_0']}, IL10={cond_row['L2_0']}")
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Observable')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_conditions, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()
