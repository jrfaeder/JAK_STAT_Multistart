# run_multistart_from_files.jl
# Loads PEtab problem from TSV files, allowing manual editing of parameters/conditions

using Pkg
using ReactionNetworkImporters, Catalyst
using DifferentialEquations, ModelingToolkit
using PEtab, DataFrames, CSV
using Optimization, Optim, OptimizationOptimJL
using SymbolicUtils, Symbolics
using Random

# ============================================================================
# CONFIGURATION - Edit paths as needed
# ============================================================================
const MODEL_NET = joinpath(@__DIR__, "variable_JAK_STAT_SOCS_degrad_model.net")
const PETAB_DIR = joinpath(@__DIR__, "petab_files")

const MEASUREMENTS_FILE = joinpath(PETAB_DIR, "measurements.tsv")
const CONDITIONS_FILE = joinpath(PETAB_DIR, "conditions.tsv")
const PARAMETERS_FILE = joinpath(PETAB_DIR, "parameters.tsv")
const OBSERVABLES_FILE = joinpath(PETAB_DIR, "observables.tsv")

# ============================================================================
# SETUP FUNCTIONS
# ============================================================================

function load_petab_from_files()
    println("Loading PEtab problem from TSV files...")
    println("  Model: $MODEL_NET")
    println("  PEtab files: $PETAB_DIR")
    
    # 1. Load the BNGL model
    prn = loadrxnetwork(BNGNetwork(), MODEL_NET)
    rsys = complete(prn.rn)
    odesys = structural_simplify(convert(ODESystem, rsys); simplify=true)
    
    # 2. Load PEtab TSV files
    measurements_df = CSV.read(MEASUREMENTS_FILE, DataFrame; delim='\t')
    conditions_df = CSV.read(CONDITIONS_FILE, DataFrame; delim='\t')
    parameters_df = CSV.read(PARAMETERS_FILE, DataFrame; delim='\t')
    observables_df = CSV.read(OBSERVABLES_FILE, DataFrame; delim='\t')
    
    println("  Loaded $(nrow(measurements_df)) measurements")
    println("  Loaded $(nrow(conditions_df)) conditions")
    println("  Loaded $(nrow(parameters_df)) parameters")
    
    # 3. Build simulation conditions dictionary
    model_params = prn.p
    param_map = Dict(string(Symbolics.getname(k)) => k for (k, v) in model_params)
    
    sim_conditions = Dict{String, Dict{Symbol, Float64}}()
    for row in eachrow(conditions_df)
        cond_id = row.conditionId
        cond_dict = Dict{Symbol, Float64}()
        for col in names(conditions_df)
            if col != "conditionId" && !ismissing(row[col])
                # Map column name to model parameter symbol
                if haskey(param_map, col)
                    cond_dict[Symbolics.getname(param_map[col])] = Float64(row[col])
                else
                    cond_dict[Symbol(col)] = Float64(row[col])
                end
            end
        end
        sim_conditions[cond_id] = cond_dict
    end
    
    # 4. Build PEtab parameters list
    petab_params = PEtabParameter[]
    for row in eachrow(parameters_df)
        p_id = row.parameterId
        p_scale = Symbol(row.parameterScale)
        p_lb = Float64(row.lowerBound)
        p_ub = Float64(row.upperBound)
        p_nominal = Float64(row.nominalValue)
        p_estimate = row.estimate == 1
        
        # Check if parameter exists in model or is an observable parameter
        if haskey(param_map, p_id)
            push!(petab_params, PEtabParameter(param_map[p_id]; 
                value=p_nominal, estimate=p_estimate, scale=p_scale, lb=p_lb, ub=p_ub))
        else
            # Observable parameter (sf_, sigma_, etc.)
            push!(petab_params, PEtabParameter(Symbol(p_id); 
                value=p_nominal, estimate=p_estimate, scale=p_scale, lb=p_lb, ub=p_ub))
        end
    end
    
    # 5. Build observables dictionary
    @parameters sf_pSTAT1 sigma_pSTAT1  # Declare observable parameters
    
    observables = Dict{String, PEtabObservable}()
    for row in eachrow(observables_df)
        obs_id = row.observableId
        
        # Find the model observable
        model_obs_sym = nothing
        obs_name = replace(row.observableFormula, r"sf_\w+ \* " => "")  # Extract base observable name
        for obs_eq in observed(rsys)
            obs_lhs_str = string(obs_eq.lhs)
            if contains(obs_lhs_str, obs_name)
                model_obs_sym = obs_eq.rhs
                break
            end
        end
        
        if isnothing(model_obs_sym)
            @warn "Could not find observable '$obs_name' in model, using placeholder"
            model_obs_sym = species(rsys)[1]
        end
        
        # Parse noise formula
        noise_param = Symbol(row.noiseFormula)
        
        observables[obs_id] = PEtabObservable(sf_pSTAT1 * model_obs_sym, noise_param)
    end
    
    # 6. Prepare measurements DataFrame for PEtab
    # Rename columns to match PEtab.jl expectations
    meas_df = copy(measurements_df)
    if hasproperty(meas_df, :simulationConditionId)
        rename!(meas_df, :simulationConditionId => :simulation_id)
    end
    
    # 7. Create PEtab model and problem
    petab_model = PEtabModel(
        odesys, observables, meas_df, petab_params;
        simulation_conditions=sim_conditions,
        verbose=true
    )
    
    petab_problem = PEtabODEProblem(petab_model)
    
    println("✅ PEtab problem loaded successfully")
    println("  Parameters to estimate: $(length(petab_problem.lower_bounds))")
    
    return petab_problem
end

# ============================================================================
# MULTISTART OPTIMIZATION
# ============================================================================

function run_multistart(n_starts=10; max_iterations=1000)
    petab_problem = load_petab_from_files()
    
    optimizer = Optim.IPNewton()
    
    # Generate start guesses
    lb = petab_problem.lower_bounds
    ub = petab_problem.upper_bounds
    n_params = length(lb)
    
    Random.seed!(1234)
    startguesses = [lb .+ rand(n_params) .* (ub .- lb) for _ in 1:n_starts]
    
    println("\n🚀 Starting multistart optimization")
    println("  Starts: $n_starts")
    println("  Parameters: $n_params")
    println("  Max iterations per start: $max_iterations")
    
    best_res = nothing
    best_f = Inf
    results = []
    
    for (i, p0) in enumerate(startguesses)
        print("Run $i/$n_starts: ")
        try
            res = calibrate(petab_problem, p0, optimizer; 
                options=Optim.Options(iterations=max_iterations, show_trace=false))
            push!(results, res)
            
            if !isnan(res.fmin) && res.fmin < best_f
                best_f = res.fmin
                best_res = res
                println("✓ NEW BEST fmin=$(round(res.fmin, digits=4))")
            else
                println("fmin=$(isnan(res.fmin) ? "NaN" : round(res.fmin, digits=4))")
            end
        catch e
            println("✗ Failed: $(typeof(e))")
        end
    end
    
    println("\n" * "="^60)
    println("OPTIMIZATION COMPLETE")
    println("="^60)
    
    if !isnothing(best_res) && !isinf(best_f)
        println("Best cost: $best_f")
        println("\nBest parameters:")
        for (i, (name, val)) in enumerate(zip(petab_problem.θ_names, best_res.xmin))
            println("  $name = $(round(val, digits=6))")
        end
        
        # Save results
        results_file = joinpath(@__DIR__, "optimization_results.csv")
        results_df = DataFrame(
            parameter = collect(petab_problem.θ_names),
            value = best_res.xmin
        )
        CSV.write(results_file, results_df)
        println("\n💾 Results saved to: $results_file")
    else
        println("⚠️  No successful optimization runs. Check model/data compatibility.")
    end
    
    return results, best_res
end

# ============================================================================
# MAIN
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # Default: 5 starts for testing, increase for production runs
    n_starts = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 5
    run_multistart(n_starts)
end
