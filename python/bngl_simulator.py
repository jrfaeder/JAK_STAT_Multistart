"""
BNGL Sensitivity Analysis Library

This module provides a general-purpose class for performing sensitivity analysis
on BNGL models using both local and global methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import bionetgen
import warnings
from typing import List, Optional, Dict, Tuple
from scipy.stats import qmc
from sklearn.linear_model import LinearRegression


class BNGLSimulator:
    """
    A class for simulating BNGL models and performing sensitivity analysis.

    Attributes:
        model: BioNetGen model object
        sim: RoadRunner simulator object
        observables: List of observable names
        parameters: List of parameter names
    """

    def __init__(self, bngl_file: str, param_values: Optional[Dict[str, float]] = None):
        """
        Initialize the simulator with a BNGL model file.

        Args:
            bngl_file: Path to the BNGL model file
            param_values: Optional dictionary of parameter values to override
                         those in the BNGL file. These values are set in the
                         model before creating the simulator, so they become
                         the "nominal" values used by resetAll().
        """
        self.model = bionetgen.bngmodel(bngl_file)

        # Set parameter values in the model before creating simulator
        # This ensures these values become the reset defaults
        if param_values:
            for param_name, value in param_values.items():
                if param_name in self.model.parameters:
                    self.model.parameters[param_name].value = value

        self.sim = self.model.setup_simulator()
        self.observables = [o for o in self.model.observables]
        self.parameters = [p for p in self.model.parameters if not p.startswith('_')]

        # Configure integrator for better tolerance
        try:
            self.sim.integrator.setValue('maximum_num_steps', 20000)
            self.sim.integrator.setValue('absolute_tolerance', 1e-12)
            self.sim.integrator.setValue('relative_tolerance', 1e-9)
        except:
            pass

    def simulate(self, t_end: float = 1200, n_steps: int = 200,
                 reset: bool = True, reference_values: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Run a simulation with current parameter values.

        Args:
            t_end: End time for simulation
            n_steps: Number of time steps
            reset: Whether to reset the simulator before running
            reference_values: Optional dictionary of parameter values to set before simulation

        Returns:
            Simulation results as a numpy array
        """
        # Set reference values first if provided
        if reference_values:
            for param_name, value in reference_values.items():
                self.set_parameter(param_name, value)

        # Reset AFTER setting reference values to ensure they take effect
        if reset:
            self.sim.reset()

        selections = ['Time'] + self.observables
        return self.sim.simulate(0, t_end, n_steps, selections=selections)

    def set_parameter(self, param_name: str, value: float):
        """
        Set a parameter value.

        Args:
            param_name: Name of the parameter
            value: New value for the parameter
        """
        self.sim[param_name] = value

    def get_parameter(self, param_name: str) -> float:
        """
        Get a parameter value.

        Args:
            param_name: Name of the parameter

        Returns:
            Current value of the parameter
        """
        return self.sim[param_name]

    def reset_all(self):
        """Reset all parameters to their original values."""
        self.sim.resetAll()

    def write_bngl(self, output_file: str, comment: Optional[str] = None):
        """
        Write a BNGL file with current parameter values.

        Args:
            output_file: Path to the output BNGL file
            comment: Optional comment to add to the file header
        """
        with open(output_file, 'w') as f:
            # Write header
            f.write("# BNGL model file generated from BNGLSimulator\n")
            if comment:
                f.write(f"# {comment}\n")
            f.write("#\n\n")

            f.write("begin model\n\n")

            # Write parameters with current values
            f.write("begin parameters\n")
            for param_name in self.parameters:
                value = self.get_parameter(param_name)
                f.write(f"  {param_name:30s} {value:.6g}\n")
            f.write("end parameters\n\n")

            # Write molecule types
            f.write("begin molecule types\n")
            for name in self.model.molecule_types:
                mol_type = self.model.molecule_types[name]
                f.write(f"  {mol_type.molecule}\n")
            f.write("end molecule types\n\n")

            # Write seed species
            f.write("begin seed species\n")
            for idx in self.model.species:
                sp = self.model.species[idx]
                f.write(f"  {sp.pattern} {sp.count}\n")
            f.write("end seed species\n\n")

            # Write observables
            f.write("begin observables\n")
            for name in self.model.observables:
                obs = self.model.observables[name]
                patterns = " ".join(str(p) for p in obs.patterns)
                f.write(f"  {obs.type} {obs.name} {patterns}\n")
            f.write("end observables\n\n")

            # Write reaction rules
            f.write("begin reaction rules\n")
            for name in self.model.rules:
                rule = self.model.rules[name]
                reactants = " + ".join(str(r) for r in rule.reactants)
                products = " + ".join(str(p) for p in rule.products)

                # Handle bidirectional rules
                if rule.bidirectional:
                    f.write(f"  {name}: {reactants} <-> {products} {rule.rate_constants[0]}, {rule.rate_constants[1]}\n")
                else:
                    f.write(f"  {name}: {reactants} -> {products} {rule.rate_constants[0]}\n")
            f.write("end reaction rules\n\n")

            f.write("end model\n\n")

            # Write actions
            f.write("# Generate network and simulate\n")
            f.write("generate_network({overwrite=>1})\n")

        print(f"BNGL file written to: {output_file}")

    def local_sensitivity_analysis(
        self,
        observables: Optional[List[str]] = None,
        parameters: Optional[List[str]] = None,
        reference_values: Optional[Dict[str, float]] = None,
        t_end: float = 600,
        n_steps: int = 200,
        perturbation: float = 1e-4
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Perform local sensitivity analysis on specified parameters.

        Args:
            observables: List of observable names to analyze (None = all observables)
            parameters: List of parameter names (None = all parameters)
            reference_values: Optional dictionary of parameter values to use as
                            reference point for sensitivity analysis. Useful for
                            setting control parameters (e.g., ligand concentrations).
                            Any parameters not in this dict use resetAll() defaults.
            t_end: End time for simulation
            n_steps: Number of time steps
            perturbation: Relative perturbation size (default 1e-4 = 0.01%)

        Returns:
            Nested dictionary: {observable: {parameter: sensitivity_array}}
        """
        if parameters is None:
            parameters = self.parameters

        if observables is None:
            observables = self.observables
        elif isinstance(observables, str):
            # Allow single observable as string for backward compatibility
            observables = [observables]

        # Validate observables
        for obs in observables:
            if obs not in self.observables:
                raise ValueError(f"Observable '{obs}' not found in model")

        # Reset to nominal values
        self.sim.resetAll()

        # Apply reference values if provided
        if reference_values:
            for param_name, value in reference_values.items():
                if param_name in self.parameters:
                    self.set_parameter(param_name, value)
                else:
                    print(f"Warning: Reference parameter '{param_name}' not found in model")

        # Compute nominal trajectory
        res_nom = self.simulate(t_end, n_steps)

        # Store nominal trajectories for all observables
        O_nom = {obs: res_nom[obs][1:] for obs in observables}  # Skip t=0

        # Store time points for reference
        self.last_time_points = res_nom['time'][1:]

        # Compute sensitivities for each parameter
        # Structure: {observable: {parameter: sensitivity_array}}
        sensitivities = {obs: {} for obs in observables}

        for param in parameters:
            if param not in self.parameters:
                print(f"Warning: Parameter '{param}' not found, skipping")
                continue

            # Reset to reference state
            self.sim.resetAll()
            if reference_values:
                for param_name, value in reference_values.items():
                    if param_name in self.parameters:
                        self.set_parameter(param_name, value)

            orig_value = self.get_parameter(param)
            self.set_parameter(param, (1 + perturbation) * orig_value)
            self.sim.reset()

            res_pert = self.simulate(t_end, n_steps, reset=False)

            # Compute sensitivities for all observables
            for obs in observables:
                O_pert = res_pert[obs][1:]

                # Compute normalized sensitivity: (dO/O) / (dp/p)
                # Use absolute sensitivity when observable is near zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    normalized_sens = ((O_pert - O_nom[obs]) / O_nom[obs]) / perturbation
                    # Replace inf/nan with absolute sensitivity scaled by parameter value
                    absolute_sens = (O_pert - O_nom[obs]) / (perturbation * orig_value)
                    normalized_sens = np.where(
                        np.isfinite(normalized_sens),
                        normalized_sens,
                        absolute_sens
                    )
                sensitivities[obs][param] = normalized_sens

        return sensitivities

    def plot_local_sensitivity(
        self,
        sensitivities: Dict[str, np.ndarray],
        observable: str,
        time_index: int = -1,
        top_n: Optional[int] = None,
        thresh: float = 0.0,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot local sensitivity results as a bar chart.

        Args:
            sensitivities: Dictionary from local_sensitivity_analysis
            observable: Name of the observable
            time_index: Index of time point to plot (-1 = final time)
            top_n: Number of most sensitive parameters to plot (None = all)
            thresh: Minimum absolute sensitivity value to include (default 0.0)
            figsize: Figure size tuple
        """
        # Filter parameters by threshold and collect values
        params_values = []
        for p in sensitivities.keys():
            val = sensitivities[p][time_index]
            if abs(val) > thresh:
                params_values.append((p, val))

        if len(params_values) == 0:
            print(f"No parameters with |sensitivity| > {thresh}")
            return

        # Sort by absolute sensitivity value
        params_values.sort(key=lambda x: abs(x[1]), reverse=True)

        # Take top_n if specified
        if top_n is not None:
            params_values = params_values[:top_n]

        # Unpack into separate lists
        params = [p for p, v in params_values]
        values = [v for p, v in params_values]

        # Color bars based on sign
        colors = ['tab:blue' if v < 0 else 'tab:red' for v in values]

        plt.figure(figsize=figsize)
        plt.bar(params, values, color=colors)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Normalized Sensitivity')

        time_val = self.last_time_points[time_index]
        title = f'Local Sensitivity of {observable} at t={time_val:.1f}s'
        if thresh > 0:
            title += f' (|S| > {thresh})'
        if top_n is not None:
            title += f' (top {len(params)})'
        plt.title(title)
        plt.axhline(0, ls='--', color='k', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_sensitivity_trajectories(
        self,
        sensitivities: Dict[str, np.ndarray],
        observable: str,
        top_n: Optional[int] = 10,
        thresh: float = 0.0,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot sensitivity trajectories over time.

        Args:
            sensitivities: Dictionary from local_sensitivity_analysis
            observable: Name of the observable
            top_n: Number of most sensitive parameters to plot (None = all)
            thresh: Minimum absolute sensitivity value to include (default 0.0)
            figsize: Figure size tuple
        """
        # Filter parameters by threshold at final time
        params_filtered = [
            p for p in sensitivities.keys()
            if abs(sensitivities[p][-1]) > thresh
        ]

        if len(params_filtered) == 0:
            print(f"No parameters with |sensitivity| > {thresh}")
            return

        # Sort filtered parameters by absolute sensitivity at final time
        params_sorted = sorted(
            params_filtered,
            key=lambda p: abs(sensitivities[p][-1]),
            reverse=True
        )

        if top_n is not None:
            params_sorted = params_sorted[:top_n]

        plt.figure(figsize=figsize)
        for param in params_sorted:
            plt.plot(self.last_time_points, sensitivities[param], label=param)

        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Sensitivity')
        title = f'Sensitivity Trajectories for {observable}'
        if thresh > 0:
            title += f' (|S| > {thresh})'
        plt.title(title)
        plt.axhline(0, ls='--', color='k', alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def global_sensitivity_analysis(
        self,
        observable: str,
        parameters: Optional[List[str]] = None,
        t_end: float = 600,
        n_steps: int = 200,
        n_samples: int = 1000,
        oom_range: float = 2.0,
        seed: int = 42
    ) -> Tuple[Dict[str, float], float, np.ndarray, np.ndarray]:
        """
        Perform global sensitivity analysis using Latin Hypercube Sampling.

        Args:
            observable: Name of the observable to analyze
            parameters: List of parameter names (None = all parameters)
            t_end: End time for simulation
            n_steps: Number of time steps
            n_samples: Number of LHS samples
            oom_range: Orders of magnitude to vary each parameter (±)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (sensitivity_coeffs, r2_score, outputs, log_samples)
        """
        if parameters is None:
            parameters = self.parameters

        if observable not in self.observables:
            raise ValueError(f"Observable '{observable}' not found in model")

        n_params = len(parameters)

        # Define log-space bounds
        log_bounds_lower = []
        log_bounds_upper = []
        nominal_values = []

        for p in parameters:
            nominal = self.get_parameter(p)
            nominal_values.append(nominal)
            if nominal > 0:
                log_bounds_lower.append(np.log10(nominal) - oom_range)
                log_bounds_upper.append(np.log10(nominal) + oom_range)
            else:
                # For zero parameters, use a small range
                log_bounds_lower.append(-6)
                log_bounds_upper.append(-2)

        log_bounds_lower = np.array(log_bounds_lower)
        log_bounds_upper = np.array(log_bounds_upper)

        print(f"Sampling {n_params} parameters using Latin Hypercube")
        print(f"Parameters: {parameters}")

        # Generate Latin Hypercube samples
        sampler = qmc.LatinHypercube(d=n_params, seed=seed)
        unit_samples = sampler.random(n=n_samples)

        # Scale to log-space bounds
        log_samples = qmc.scale(unit_samples, l_bounds=log_bounds_lower, u_bounds=log_bounds_upper)
        param_samples = 10**log_samples

        print(f"Generated {n_samples} samples")
        print(f"Sample quality (discrepancy): {qmc.discrepancy(unit_samples):.4f}")

        # Run simulations
        outputs = []
        failed_indices = []
        selections = ['Time'] + self.observables

        print("\nRunning simulations...")
        for i, params in enumerate(param_samples):
            try:
                # Set parameter values
                for j, pname in enumerate(parameters):
                    self.set_parameter(pname, params[j])

                # Reset and simulate
                self.sim.reset()

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = self.sim.simulate(0, t_end, n_steps, selections=selections)

                output_val = res[observable][-1]

                if np.isnan(output_val) or np.isinf(output_val):
                    outputs.append(np.nan)
                    failed_indices.append(i)
                else:
                    outputs.append(output_val)

            except Exception as e:
                outputs.append(np.nan)
                failed_indices.append(i)
                if len(failed_indices) <= 5:
                    print(f"  Simulation {i} failed: {type(e).__name__}")

            if (i + 1) % 100 == 0:
                print(f"  Completed {i+1}/{n_samples} simulations ({len(failed_indices)} failed so far)")

        outputs = np.array(outputs)

        # Report statistics
        n_failed = len(failed_indices)
        n_success = n_samples - n_failed
        print(f"\nSimulation complete!")
        print(f"  Successful: {n_success}/{n_samples} ({100*n_success/n_samples:.1f}%)")
        print(f"  Failed: {n_failed}/{n_samples} ({100*n_failed/n_samples:.1f}%)")

        # Filter out failed simulations
        valid_mask = ~np.isnan(outputs)
        valid_outputs = outputs[valid_mask]
        valid_log_samples = log_samples[valid_mask]

        print(f"\nOutput range (valid): [{valid_outputs.min():.2e}, {valid_outputs.max():.2e}]")

        # Perform linear regression in log space
        X_log = valid_log_samples
        y_log = np.log10(valid_outputs + 1e-10)

        reg = LinearRegression()
        reg.fit(X_log, y_log)

        sensitivity_coeffs = dict(zip(parameters, reg.coef_))
        r2_score = reg.score(X_log, y_log)

        print(f"\nR² score: {r2_score:.4f}")
        print(f"\nGlobal sensitivity coefficients:")
        for pname, coeff in sensitivity_coeffs.items():
            print(f"  {pname:12s}: {coeff:+.4f}")

        # Reset simulator to nominal values
        self.reset_all()

        return sensitivity_coeffs, r2_score, outputs, log_samples

    def plot_global_sensitivity(
        self,
        sensitivity_coeffs: Dict[str, float],
        r2_score: float,
        observable: str,
        outputs: np.ndarray,
        log_samples: np.ndarray,
        t_end: float,
        thresh: float = 0.0,
        figsize: Tuple[int, int] = (14, 5)
    ):
        """
        Plot global sensitivity results.

        Args:
            sensitivity_coeffs: Dictionary of sensitivity coefficients
            r2_score: R² score from regression
            observable: Name of the observable
            outputs: Array of simulation outputs
            log_samples: Array of log-space parameter samples
            t_end: End time of simulations
            thresh: Minimum absolute sensitivity coefficient to include (default 0.0)
            figsize: Figure size tuple
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Left plot: Sensitivity coefficients (filtered by threshold)
        params = []
        coeffs = []
        for p, c in sensitivity_coeffs.items():
            if abs(c) > thresh:
                params.append(p)
                coeffs.append(c)

        if len(params) == 0:
            print(f"No parameters with |sensitivity| > {thresh}")
            return

        colors = ['tab:blue' if c < 0 else 'tab:red' for c in coeffs]

        ax1.bar(params, coeffs, color=colors)
        ax1.set_xticklabels(params, rotation=45, ha='right')
        ax1.set_ylabel('Sensitivity Coefficient')
        title = f'Global Sensitivity of {observable} at t={t_end}s'
        if thresh > 0:
            title += f'\n(|S| > {thresh})'
        ax1.set_title(title)
        ax1.axhline(0, ls='--', color='k', alpha=0.3)

        # Right plot: Predicted vs Actual
        valid_mask = ~np.isnan(outputs)
        valid_outputs = outputs[valid_mask]
        valid_log_samples = log_samples[valid_mask]

        X_log = valid_log_samples
        y_log = np.log10(valid_outputs + 1e-10)

        reg = LinearRegression()
        reg.fit(X_log, y_log)
        y_pred = reg.predict(X_log)

        ax2.scatter(y_log, y_pred, alpha=0.5, s=30)
        min_val = min(y_log.min(), y_pred.min())
        max_val = max(y_log.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        ax2.set_xlabel(f'Actual log₁₀({observable})')
        ax2.set_ylabel(f'Predicted log₁₀({observable})')
        ax2.set_title(f'Prediction Accuracy (R²={r2_score:.3f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()