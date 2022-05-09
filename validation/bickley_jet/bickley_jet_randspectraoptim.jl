# cd("/home/brynn/Code/Oceananigans.jl")
cd("/home/ameza/batou-home/Repos/Oceananigans.jl")
#first you need to add the EKP package from your github repo
#using Pkg; Pkg.add(url = "/Users/anthonymeza/Documents/GitHub/EnsembleKalmanProcesses.jl/")
# using Pkg; Pkg.add(url = "/home/ameza/batou-home/Repos/EnsembleKalmanProcesses.jl/")
using Pkg; Pkg.activate(".")
# Pkg.instantiate()
using Pkg, FileIO
using Oceananigans
using Oceananigans.Units
using Oceananigans.Advection: VelocityStencil, VorticityStencil
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.DataContainers
using EnsembleKalmanProcesses.ParameterDistributions
using Random
using LinearAlgebra
using Statistics
using Distributions
const EKP = EnsembleKalmanProcesses 
using Printf
using GLMakie
using JLD2

cd("validation/bickley_jet")
include("bickley_utils.jl")
exp_name = "G3_loss"
mkpath(exp_name)
println("Running experiment with... ", exp_name)

N_iterations = 5
N_ens = 5
rng_seed = 4137
Random.seed!(rng_seed)

dim_output = 1
stabilization_level = 1e-3
Γ_stabilization = stabilization_level * Matrix(I, dim_output, dim_output)

G_target = [0]

#prior_list = Vector{Dict{String,Any}}
prior_list = []
for a_i in 1:36
    temp = Dict("distribution" => Parameterized(Normal(5, 1)), "constraint" => no_constraint(), "name" => "u"*string(a_i) ) 
    push!(prior_list, temp)
end
prior_list = convert(Vector{Dict{String,Any}}, prior_list)
priors = EKP.ParameterDistribution(prior_list)
initial_ensemble = EKP.construct_initial_ensemble(priors, N_ens; rng_seed = rng_seed)
ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, G_target, Γ_stabilization, Inversion())

function G1(name)
    filepath = name * ".jld2"
    ζt = FieldTimeSeries(filepath, "ζ")
    t = ζt.times
    Nt = length(t)
    z_start = interior(ζt[1], :, :, 1)
    z_end = interior(ζt[Nt], :, :, 1)
    enst_start = sum(z_start.^2)
    enst_end = sum(z_end.^2)

    spec_2048 = [24.312371182474553, 0.016955630982388614, 0.006849002562262672, 0.003360390720253825]

    include("post_process_spec.jl")
    spec = hov_ζ_w1_128_x[2:4]
    
    return (enst_start - enst_end)^2
end

function G2(name)
    filepath = name * ".jld2"
    ζt = FieldTimeSeries(filepath, "ζ")
    t = ζt.times
    Nt = length(t)
    z_start = [sum(interior(ζt[i], :, :, 1).^2) for i = 1:Nt-1]
    z_end = [sum(interior(ζt[i], :, :, 1).^2) for i = 2:Nt]
    # enst = zeros(length(z_start))
    temp1 = sum((z_end .- z_start).^2)/length(z_end)
    return sqrt(temp1)
end

function G3(name)
    #
    spec_2048 = log10.([0.016955630982388614, 0.006849002562262672, 0.003360390720253825])

    include("post_process_spec.jl")
    spec = log10.(hov_ζ_w1_128_x[2:4])
    
    return sum((spec .- spec_2048).^2)
end

save_losses = zeros(N_ens, N_iterations)

"""
    run_bickley_jet(output_time_interval = 2, stop_time = 200, arch = CPU(), Nh = 64, ν = 0,
                    momentum_advection = VectorInvariant())

Run the Bickley jet validation experiment until `stop_time` using `momentum_advection`
scheme or formulation, with horizontal resolution `Nh`, viscosity `ν`, on `arch`itecture.
"""
function run_bickley_jet(thread_id;
                         output_time_interval = 2,
                         stop_time = 100,
                         arch = CPU(),
                         Nh = 64, 
                         free_surface = ImplicitFreeSurface(gravitational_acceleration=10.0),
                         momentum_advection = WENO5(),
                         tracer_advection = WENO5(),
                         experiment_name = string(nameof(typeof(momentum_advection))))

    grid = bickley_grid(; arch, Nh)
    model = HydrostaticFreeSurfaceModel(; grid, momentum_advection, tracer_advection,
                                        free_surface, tracers = :c, buoyancy=nothing)
    set_bickley_jet!(model)

    Δt = 0.2 * 2π / Nh
    wizard = TimeStepWizard(cfl=0.2, max_change=1.1, max_Δt=10.0)
    simulation = Simulation(model; Δt, stop_time)

    progress(sim) = @printf("Iter: %d, time: %.1f, Δt: %.1e, max|u|: %.3f, max|η|: %.3f\n",
                            iteration(sim), time(sim), sim.Δt,
                            maximum(abs, model.velocities.u),
                            maximum(abs, model.free_surface.η))

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))
    
    wizard = TimeStepWizard(cfl=0.2, max_change=1.1, max_Δt=10.0)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

    # Output: primitive fields + computations
    u, v, w = model.velocities
    outputs = merge(model.velocities, model.tracers, (ζ=∂x(v) - ∂y(u), η=model.free_surface.η))

    @show output_name = "bickley_jet_Nh_$(Nh)_" * experiment_name * exp_name * string(thread_id)

    simulation.output_writers[:fields] =
        JLD2OutputWriter(model, outputs,
                                schedule = TimeInterval(output_time_interval),
                                filename = output_name,
                                overwrite_existing = true)

    @info "Running a simulation of an unstable Bickley jet with $(Nh)² degrees of freedom..."

    start_time = time_ns()

    run!(simulation)

    elapsed = 1e-9 * (time_ns() - start_time)
    @info "... the bickley jet simulation took " * prettytime(elapsed)

    return output_name
end
    
"""
    visualize_bickley_jet(experiment_name)

Visualize the Bickley jet data in `name * ".jld2"`.
"""

function visualize_bickley_jet(name)
    @info "Making a fun movie about an unstable Bickley jet..."

    filepath = name * ".jld2"

    ζt = FieldTimeSeries(filepath, "ζ")
    ct = FieldTimeSeries(filepath, "c")
    t = ζt.times
    Nt = length(t)

    fig = GLMakie.Figure(resolution=(1400, 800))
    slider = GLMakie.Slider(fig[2, 1:2], range=1:Nt, startvalue=1)
    n = slider.value

    ζtitle = GLMakie.@lift @sprintf("ζ at t = %.1f", t[$n])
    ctitle = GLMakie.@lift @sprintf("c at t = %.1f", t[$n])

    ax_ζ = GLMakie.Axis(fig[1, 1], title=ζtitle, aspect=1)
    ax_c = GLMakie.Axis(fig[1, 2], title=ctitle, aspect=1)

    ζ = GLMakie.@lift interior(ζt[$n], :, :, 1)
    c = GLMakie.@lift interior(ct[$n], :, :, 1)

    GLMakie.heatmap!(ax_ζ, ζ, colorrange=(-1, 1), colormap=:redblue)
    GLMakie.heatmap!(ax_c, c, colorrange=(-1, 1), colormap=:thermal)

    GLMakie.record(fig, name * ".mp4", 1:Nt, framerate=24) do nn
        @info "Drawing frame $nn of $Nt..."
        n[] = nn
    end
end


arch = CPU()
for i in 1:N_iterations     
    params_i = get_u_final(ensemble_kalman_process)      
    g_ens = zeros(N_ens)
    for j in 1:N_ens
        params = params_i[:, j] 
        coeffs = Tuple(Tuple(params[k + (l*6)] for k in 1:6) for l in 0:5)
        momentum_advection = WENO5(vector_invariant = VorticityStencil(), smoothness_coeffs = coeffs)
        Nh = 128       
        name = run_bickley_jet(Threads.threadid(); arch, momentum_advection, Nh)
        g_ens[j] = G3(name)
        #data = generate_spectra(name) #DO THIS !!!
    end
    save_losses[:, i] .= g_ens
    println(g_ens)
    g_ens = convert(Matrix{Float64}, g_ens')
    EKP.update_ensemble!(ensemble_kalman_process, g_ens)
end

save_object(exp_name*"/"*exp_name*"_losses.jld2", save_losses)
u_init = get_u_prior(ensemble_kalman_process)
u_final = get_u_final(ensemble_kalman_process)

β_optim = mean(u_final, dims = 2)
coeffs = Tuple(Tuple(β_optim[k*l] for k in 1:6) for l in 1:6)
momentum_advection = WENO5(vector_invariant = VorticityStencil(), smoothness_coeffs = coeffs)
Nh = 128        
name = run_bickley_jet(0; arch, momentum_advection, Nh)
# visualize_bickley_jet(name)
