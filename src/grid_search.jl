using Dates

# Import the main training code
include("gpt.jl")

# Define the hyperparameter grid
const GRID = Dict(
    "n_embed" => [32, 64, 128],
    "n_hidden" => [128, 256, 512],
    "n_heads" => [4, 8],
    "n_layers" => [4, 6, 8],
    "dropout" => [0.0, 0.1, 0.2],
    "lr" => [1e-3, 1e-2, 5e-2]
)

# Create a timestamp for this run
const TIMESTAMP = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
const RESULTS_DIR = "grid_search_results_$(TIMESTAMP)"
mkpath(RESULTS_DIR)

# Function to generate all combinations of hyperparameters
function generate_combinations(grid)
    param_names = collect(keys(grid))
    param_values = collect(values(grid))
    combinations = []
    
    function generate_recursive(current, depth)
        if depth > length(param_names)
            push!(combinations, Dict(param_names[i] => current[i] for i in 1:length(param_names)))
            return
        end
        for v in param_values[depth]
            current[depth] = v
            generate_recursive(current, depth + 1)
        end
    end
    
    generate_recursive(Array{Any}(undef, length(param_names)), 1)
    return combinations
end

# Main grid search function
function run_grid_search()
    # Generate all combinations
    combinations = generate_combinations(GRID)
    total_runs = length(combinations)
    
    println("Starting grid search with $total_runs combinations")
    
    # Create a summary file for all runs
    summary_file = joinpath(RESULTS_DIR, "summary.txt")
    open(summary_file, "w") do io
        println(io, "# run_id n_embed n_hidden n_heads n_layers dropout lr train_loss test_loss")
    end
    
    for (run_id, config) in enumerate(combinations)
        println("\nRun $run_id/$total_runs")
        println("Configuration: ", config)
                
        # Run training with current config
        args, model, test_loss, train_loss = train(; Dict(Symbol(k) => v for (k,v) in config)...)

        # Append to summary file
        open(summary_file, "a") do io
            println(io, "$run_id $(config["n_embed"]) $(config["n_hidden"]) $(config["n_heads"]) $(config["n_layers"]) $(config["dropout"]) $(config["lr"]) $train_loss $test_loss")
        end
    end

end

# Run the grid search if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_grid_search()
end

