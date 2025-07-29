using ArgParse

# Global variables for REPL usage
global model = nothing

function main(args)
    s = ArgParseSettings(description="Innisvouls: GPT and LSTM model for sequence generation")
    
    @add_arg_table! s begin
        "--train"
            help = "Train a new model"
            action = :store_true
        "--epochs"
            help = "Number of epochs to train for"
            arg_type = Int
            default = 20
        "--load"
            help = "Filename of model to load from checkpoint"
            arg_type = String
            default = "model-checkpoint.jld2"
        "--textseed"
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
        "-N"
            help = "Number of FASTA sequences to generate"
            arg_type = Int
            default = 10
    end
    parsed_args = parse_args(s)
    
    if parsed_args["train"]
        global args,model = train()
    else
        println("Loading model from checkpoint... (many errors may occur here!)")
        global args, model = load_model(parsed_args["load"]) |> device
    end
    
    if !isnothing(model)
        generate_FASTA(model, parsed_args["generate"], parsed_args["length"], temperature=parsed_args["temperature"], N=parsed_args["N"])
    end
end

function generate_FASTA(model, seed, length; temperature=1.0, N=10)
    for i in 1:N
        println(">seq$i")
        seq=generate(model, seed, length, temperature=temperature)
        words=split(seq, '_') 
        println(words[2])
        println()
    end
end


# For REPL usage:
# Set args = Args(n_embed=64, n_hidden=256, ...) to customize parameters
# Then call train() or load_model("model-checkpoint.jld2") to initialize the model
# Use generate(model, seed, length, temperature=t) to generate text 



