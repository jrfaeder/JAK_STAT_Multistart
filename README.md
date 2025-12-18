# Neha Multistart Optimization

This repository contains the codebase for running multistart parameter estimation and identifiability analysis for the JAK-STAT-SOCS degradation model.

## System Overview

The pipeline consists of three main stages, designed to run on a SLURM cluster:
1.  **Optimization (Array Job)**: Runs hundreds of independent optimization fits in parallel.
2.  **Collation**: Aggregates the results from all fits to find the best parameters.
3.  **Identifiability**: Performs structural identifiability analysis on the model.

## Configuration

Before running any jobs, you must create a local configuration file to define your environment.

1.  Create `user_config_slurm.sh` in the root directory (this file is ignored by git).
2.  Add your email and path definitions:

```bash
#!/bin/bash
export EMAIL="abc@email.com"
export PROJECT_HOME="/path/to/project"
export IL6_HOME="/path/to/IL6_TGFB"  # Path to dependencies/sysimage
```

## Execution

Scripts are located in the `scripts/` directory. You can submit them using `sbatch` or the provided helper.

### 1. Run Optimization Array
This submits a job array (default 100 tasks) to fit the model from different starting points.

```bash
# Load config
source user_config_slurm.sh

# Submit
sbatch --mail-user="$EMAIL" scripts/submit_array.sh
```

### 2. Collate Results
After the array job finishes, run this to aggregate results and generate plots.

```bash
sbatch --mail-user="$EMAIL" scripts/submit_collate.sh
```

### 3. Identifiability Analysis
Run this to perform structural identifiability analysis on the best fit.

```bash
sbatch --mail-user="$EMAIL" scripts/submit_identifiability.sh
```

**Resource Usage**: 1 CPU, 31GB Memory.

## Directory Structure

-   `scripts/`: SLURM submission scripts.
-   `run_single_task.jl`: Main Julia script for a single optimization run.
-   `collect_results.jl`: Script for aggregating results and plotting.
-   `structural_identifiability.jl`: Script for identifiability analysis.
-   `petab_files/`: PEtab configuration files (measurements, parameters, etc.).
-   `results/`: (Generated) individual fit results.
-   `final_results_plots/`: (Generated) summary plots and CSVs.
