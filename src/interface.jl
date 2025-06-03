using ArgParse
using .GPT

# Global variables for REPL usage
global args = Args()
global model = nothing

function parse_commandline()
    s = ArgParseSettings(description="GPT Model Interface")
    
    @add_arg_table! s begin
        "--train"
            help = "Train a new model"
            action = :store_true
        "--load"
            help = "Load model from checkpoint"
            arg_type = String
            default = "model-checkpoint.jld2"
        "--generate"
            help = "Generate text with given seed"
            arg_type = String
            default = "_"
        "--length"
            help = "Length of generated text"
            arg_type = Int
            default = 80
        "--temperature"
            help = "Temperature for generation"
            arg_type = Float64
            default = 1.0
    end
    
    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    
    if parsed_args["train"]
        global args, model = train()
    else
        global args, model = load_model(parsed_args["load"]) |> device
    end
    
    if !isnothing(model)
        println(generate(model, parsed_args["generate"], parsed_args["length"], temperature=parsed_args["temperature"]))
    end
end

# For REPL usage:
# Set args = Args(n_embed=64, n_hidden=256, ...) to customize parameters
# Then call train() or load_model("model-checkpoint.jld2") to initialize the model
# Use generate(model, seed, length, temperature=t) to generate text 