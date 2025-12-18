# Script to export and inspect PEtab problem components
using Pkg
using ReactionNetworkImporters, Catalyst
using DifferentialEquations, ModelingToolkit
using PEtab, DataFrames, XLSX, CSV
using SymbolicUtils, Symbolics

const MODEL_NET = joinpath(@__DIR__, "variable_JAK_STAT_SOCS_degrad_model.net")
const DATA_PATH = joinpath(@__DIR__, "Data", "pSTAT1_pooled_data.xlsx")

function parse_condition(col_name)
    l1_val = 0.0
    l2_val = 0.0
    parts = split(col_name, "_")
    i = 1
    while i <= length(parts)
        if parts[i] == "il6"
            val_str = parts[i+1]
            l1_val = val_str == "01" ? 0.1 : parse(Float64, val_str)
            i += 2
        elseif parts[i] == "il10"
            val_str = parts[i+1]
            l2_val = val_str == "01" ? 0.1 : parse(Float64, val_str)
            i += 2
        else
            i += 1
        end
    end
    return l1_val, l2_val
end

function export_petab_tables()
    println("Loading model and data...")
    
    prn = loadrxnetwork(BNGNetwork(), MODEL_NET)
    rsys = complete(prn.rn)
    
    xf = XLSX.readxlsx(DATA_PATH)
    sheet = xf[1]
    df = DataFrame(XLSX.gettable(sheet))
    
    time_col = nothing
    for n in names(df)
        if lowercase(n) == "time"
            time_col = n
            break
        end
    end
    
    model_params = prn.p
    param_map = Dict(string(Symbolics.getname(k)) => k for (k, v) in model_params)
    l1_sym = param_map["L1_0"]
    l2_sym = param_map["L2_0"]
    
    # === Build Measurement Table ===
    meas_rows = []
    conditions_rows = []
    obs_id = "obs_total_pS1"
    
    for col in names(df)
        if lowercase(col) == "time" || startswith(col, "Date")
            continue
        end
        l1_val, l2_val = parse_condition(col)
        cond_id = "cond_" * col
        push!(conditions_rows, (conditionId = cond_id, L1_0 = l1_val, L2_0 = l2_val))
    end
    # Pass 1: Find baselines (t=0) for each condition
    # Matching experimental data processing: shift = df - df.iloc[0,:]
    baselines = Dict{String, Float64}()
    for col in names(df)
        if lowercase(col) == "time" || startswith(col, "Date") continue end
        for row in eachrow(df)
            if row[time_col] == 0 && !ismissing(row[col])
                baselines[col] = Float64(row[col])
                break
            end
        end
    end
    
    # Pass 2: Find Normalization Factor (IL6=10, IL10=0 at t=20, AFTER shift)
    # Matching: pSTAT_shift_list[i].loc[20]['il6_10']
    norm_factor = 1.0
    found_ref = false
    for col in names(df)
        if lowercase(col) == "time" || startswith(col, "Date") continue end
        l1_val, l2_val = parse_condition(col)
        
        # Check for Reference Condition: IL6=10 (or 10.0), IL10=0
        if l1_val == 10.0 && l2_val == 0.0
            for row in eachrow(df)
                if row[time_col] == 20.0 && !ismissing(row[col])
                    raw_val = Float64(row[col])
                    bg = get(baselines, col, 0.0)
                    norm_factor = raw_val - bg  # SHIFTED value
                    println("  Found Normalization Ref ($col @ 20min): Raw=$raw_val, BG=$bg, NormFactor=$norm_factor")
                    found_ref = true
                    break
                end
            end
        end
        if found_ref break end
    end
    
    if !found_ref
        println("  ⚠️ Warning: Could not find reference condition (IL6=10, IL10=0 @ 20min). Using NormFactor=1.0")
    end

    # Pass 3: Generate Measurements (Shifted & Normalized)
    # Matching: (df - df.iloc[0,:]) / pSTAT_shift_list[i].loc[20]['il6_10']
    for col in names(df)
        if lowercase(col) == "time" || startswith(col, "Date") continue end
        
        cond_id = "cond_" * col
        bg = get(baselines, col, 0.0)
        
        for row in eachrow(df)
            t = row[time_col]
            val = row[col]
            if !ismissing(val)
                # Apply: (Value - Baseline) / NormFactor
                shifted_val = Float64(val) - bg
                # Match Python: set negative values to 0
                if shifted_val < 0
                    shifted_val = 0.0
                end
                final_val = shifted_val / norm_factor
                
                push!(meas_rows, (
                    observableId = obs_id,
                    simulationConditionId = cond_id,
                    time = Float64(t),
                    measurement = final_val,
                    noiseParameters = "sigma_pSTAT1"
                ))
            end
        end
    end
    
    measurements_df = DataFrame(meas_rows)
    conditions_df = DataFrame(conditions_rows) |> unique
    
    # === Build Parameters Table ===
    param_rows = []
    exclude_params = ["L1_0", "L2_0", "SOCS3_0", "SOCS1_0"]
    
    for (p_sym, val) in model_params
        p_name = string(Symbolics.getname(p_sym))
        if p_name in exclude_params
            continue
        end
        push!(param_rows, (
            parameterId = p_name,
            parameterScale = "log10",
            lowerBound = 1e-6,
            upperBound = 1e4,
            nominalValue = Float64(val),
            estimate = 1
        ))
    end
    
    # Add observable parameters
    push!(param_rows, (parameterId="sf_pSTAT1", parameterScale="log10", lowerBound=1e-7, upperBound=1e3, nominalValue=1.0, estimate=1))
    # sigma_pSTAT1 fixed at 0.15 (15% relative noise) as per paper
    push!(param_rows, (parameterId="sigma_pSTAT1", parameterScale="log10", lowerBound=1e-2, upperBound=1.0, nominalValue=0.15, estimate=0))
    
    parameters_df = DataFrame(param_rows)
    
    # === Build Observables Table ===
    # Using a relative noise model: sigma * (pred + floor)
    # This approximates the 15% CV mentioned in the paper.
    observables_df = DataFrame(
        observableId = [obs_id],
        observableFormula = ["sf_pSTAT1 * total_pS1"],
        noiseFormula = ["sigma_pSTAT1 * (sf_pSTAT1 * total_pS1 + 0.01)"]
    )
    
    # === Export to CSV ===
    output_dir = joinpath(@__DIR__, "petab_files")
    mkpath(output_dir)
    
    CSV.write(joinpath(output_dir, "measurements.tsv"), measurements_df; delim='\t')
    CSV.write(joinpath(output_dir, "conditions.tsv"), conditions_df; delim='\t')
    CSV.write(joinpath(output_dir, "parameters.tsv"), parameters_df; delim='\t')
    CSV.write(joinpath(output_dir, "observables.tsv"), observables_df; delim='\t')
    
    println("\n=== MEASUREMENT TABLE (first 10 rows) ===")
    println(first(measurements_df, 10))
    
    println("\n=== CONDITIONS TABLE ===")
    println(conditions_df)
    
    println("\n=== PARAMETERS TABLE (first 15 rows) ===")
    println(first(parameters_df, 15))
    
    println("\n=== OBSERVABLES TABLE ===")
    println(observables_df)
    
    println("\n✅ PEtab tables exported to: $output_dir")
end

export_petab_tables()
