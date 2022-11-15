using Random
using SpeedyWeather

include("modified_init.jl")

function datagen(location, n_samples, seed)
    @info "data" location n_samples seed
    Random.seed!(seed)
    loc = joinpath(location, "seed=$seed")
    mkpath(loc)
    map(_->run_speedy(Float32; n_days=20, model=:shallowwater, output=true, trunc=62, Δt_at_T85=40, initial_conditions=:random2, out_path=loc), 1:n_samples)
end

if abspath(PROGRAM_FILE) == @__FILE__
    @info "Running datagen"
    datagen(ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3]))
end
