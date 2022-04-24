using Oceananigans
using Oceananigans.Advection: VelocityStencil
using Oceananigans.MultiRegion: reconstruct_global_field
using GLMakie

grid = RectilinearGrid(size=(128, 128, 1), halo=(4, 4, 4), x=(0, 2π), y=(0, 2π), z=(0, 1), topology=(Periodic, Periodic, Bounded))
#grid = MultiRegionGrid(grid, partition=XPartition(2))

momentum_advection = WENO5() #vector_invariant=VelocityStencil())

model = HydrostaticFreeSurfaceModel(; grid, momentum_advection,
                                    tracers = (),
                                    buoyancy = nothing,
                                    free_surface = ExplicitFreeSurface(gravitational_acceleration=1),
                                    closure = ScalarDiffusivity(ν=1e-4))


ϵ(x, y, z)  =  2rand() - 1
set!(model, u=ϵ, v=ϵ)

Δh = 2π / grid.Nx
Δt = 0.1 * Δh
simulation = Simulation(model; Δt, stop_iteration=10)
run!(simulation)

simulation.stop_iteration += 1000

progress(sim) = @info "Iteration: $(iteration(sim)), time: $(time(sim))"
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

start_time = time_ns()
run!(simulation)
elapsed_time = 1e-9 * (time_ns() - start_time)
@info "Simulation ran for " * prettytime(elapsed_time)

u, v, w = model.velocities

# @show u

if grid isa MultiRegionGrid
    u = reconstruct_global_field(u)
    v = reconstruct_global_field(v)
    # @show u
end

ζ = compute!(Field(∂x(v) - ∂y(u)))

fig = Figure()
ax = Axis(fig[1, 1])
heatmap!(ax, interior(ζ, :, :, 1))
display(fig)
