# # Plankton mixing and blooming
#
# In this example, we simulate the mixing of phytoplankton by convection
# that decreases in time and eventually shuts off, thereby precipitating a
# phytoplankton bloom. A similar scenario was simulated by
# [Taylor and Ferrari (2011)](https://aslopubs.onlinelibrary.wiley.com/doi/abs/10.4319/lo.2011.56.6.2293),
# providing evidence that the
# ["critical turbulence hypothesis"](https://en.wikipedia.org/wiki/Critical_depth#Critical_Turbulence_Hypothesis)
# explains the explosive bloom of oceanic phytoplankton
# observed in spring.
#
# The phytoplankton in our model are advected, diffuse, grow, and die according to
#
# ```math
# ∂_t P + \boldsymbol{v ⋅ ∇} P - κ ∇²P = (μ₀ \exp(z / λ) - m) \, P \, ,
# ```
#
# where ``\boldsymbol{v}`` is the turbulent velocity field, ``κ`` is an isotropic diffusivity,
#  ``μ₀`` is the phytoplankton growth rate at the surface, ``λ`` is the scale over
# which sunlight attenuates away from the surface, and ``m`` is the mortality rate
# of phytoplankton due to viruses and grazing by zooplankton. We use Oceananigans'
#  `Forcing` abstraction to implement the phytoplankton dynamics described by the right
# side of the phytoplankton equation above.
#
# This example demonstrates
#
#   * How to use a user-defined forcing function to
#     simulate the dynamics of phytoplankton growth in sunlight
#     and grazing by zooplankton.
#   * How to set time-dependent boundary conditions.
#   * How to use the `TimeStepWizard` to adapt the simulation time-step.
#   * How to use `Average` to diagnose spatial averages of model fields.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, Plots, JLD2, Measures"
# ```

# ## The grid
#
# We use a two-dimensional grid with 64² points, 3² halo points for high-order advection,
# 1 m grid spacing, and a `Flat` `y`-direction:

using Oceananigans
using Oceananigans.Units: minutes, hour, hours, day

grid = RectilinearGrid(size=(64, 64), extent=(64, 64), halo=(3, 3), topology=(Periodic, Flat, Bounded))

# ## Boundary conditions
#
# We impose a surface buoyancy flux that's initially constant and then decays to zero,

buoyancy_flux(x, y, t, params) = params.initial_buoyancy_flux * exp(-t^4 / (24 * params.shut_off_time^4))

buoyancy_flux_parameters = (initial_buoyancy_flux = 1e-8, # m² s⁻³
                                    shut_off_time = 2hours)

buoyancy_flux_bc = FluxBoundaryCondition(buoyancy_flux, parameters = buoyancy_flux_parameters)

# The fourth power in the argument of `exp` above helps keep the buoyancy flux relatively
# constant during the first phase of the simulation. We produce a plot of this time-dependent
# buoyancy flux for the visually-oriented,

using Plots, Measures

times = range(0, 12hours, length=100)

flux_plot = plot(times ./ hour, [buoyancy_flux(0, 0, t, buoyancy_flux_parameters) for t in times],
                 linewidth = 2, xlabel = "Time (hours)", ylabel = "Surface buoyancy flux (m² s⁻³)",
                 size = (800, 300), margin = 5mm, label = nothing)

# The buoyancy flux effectively shuts off after 6 hours of simulation time.
#
# !!! info "The flux convention in Oceananigans.jl"
#     Fluxes are defined by the direction a quantity is carried: _positive_ velocities
#     produce _positive_ fluxes, while _negative_ velocities produce _negative_ fluxes.
#     Diffusive fluxes are defined with the same convention. A positive flux at the _top_
#     boundary transports buoyancy _upwards, out of the domain_. This means that a positive
#     flux of buoyancy at the top boundary reduces the buoyancy of near-surface fluid,
#     causing convection.
#
# The initial condition and bottom boundary condition impose the constant buoyancy gradient

N² = 1e-4 # s⁻²

buoyancy_gradient_bc = GradientBoundaryCondition(N²)

# In summary, the buoyancy boundary conditions impose a destabilizing flux
# at the top and a stable buoyancy gradient at the bottom:

buoyancy_bcs = FieldBoundaryConditions(top = buoyancy_flux_bc, bottom = buoyancy_gradient_bc)

# ## Phytoplankton dynamics: light-dependent growth and uniform mortality
#
# We use a simple model for the growth of phytoplankton in sunlight and decay
# due to viruses and grazing by zooplankton,

growing_and_grazing(x, y, z, t, P, params) = (params.μ₀ * exp(z / params.λ) - params.m) * P
nothing # hide

# with parameters

plankton_dynamics_parameters = (μ₀ = 1/day,   # surface growth rate
                                 λ = 5,       # sunlight attenuation length scale (m)
                                 m = 0.1/day) # mortality rate due to virus and zooplankton grazing

# We tell `Forcing` that our plankton model depends
# on the plankton concentration `P` and the chosen parameters,

plankton_dynamics = Forcing(growing_and_grazing, field_dependencies = :P,
                            parameters = plankton_dynamics_parameters)

# ## The model
#
# The name "`P`" for phytoplankton is specified in the
# constructor for `NonhydrostaticModel`. We additionally specify a fifth-order
# advection scheme, third-order Runge-Kutta time-stepping, isotropic viscosity and diffusivities,
# and Coriolis forces appropriate for planktonic convection at mid-latitudes on Earth.

model = NonhydrostaticModel(
                   grid = grid,
              advection = UpwindBiasedFifthOrder(),
            timestepper = :RungeKutta3,
                closure = ScalarDiffusivity(ν=1e-4, κ=1e-4),
               coriolis = FPlane(f=1e-4),
                tracers = (:b, :P), # P for Plankton
               buoyancy = BuoyancyTracer(),
                forcing = (P=plankton_dynamics,),
    boundary_conditions = (b=buoyancy_bcs,)
)

# ## Initial condition
#
# We set the initial phytoplankton at ``P = 1 \, \rm{μM}``.
# For buoyancy, we use a stratification that's mixed near the surface and
# linearly stratified below, superposed with surface-concentrated random noise.

mixed_layer_depth = 32 # m

stratification(z) = z < -mixed_layer_depth ? N² * z : - N² * mixed_layer_depth

noise(z) = 1e-4 * N² * grid.Lz * randn() * exp(z / 4)

initial_buoyancy(x, y, z) = stratification(z) + noise(z)

set!(model, b=initial_buoyancy, P=1)

# ## Simulation with adaptive time-stepping, logging, and output
#
# We build a simulation

simulation = Simulation(model, Δt=2minutes, stop_time=24hours)

# with a `TimeStepWizard` that limits the
# time-step to 2 minutes, and adapts the time-step such that CFL
# (Courant-Freidrichs-Lewy) number hovers around `1.0`,

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=2minutes)

simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# We also add a callback that prints the progress of the simulation,

using Printf

progress(sim) = @printf("Iteration: %d, time: %s, Δt: %s\n",
                        iteration(sim), prettytime(time(sim)), prettytime(sim.Δt))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))

# and a basic `JLD2OutputWriter` that writes velocities and both
# the two-dimensional and horizontally-averaged plankton concentration,

outputs = (w = model.velocities.w,
           P = model.tracers.P,
           P_avg = Average(model.tracers.P, dims=(1, 2)))

simulation.output_writers[:simple_output] =
    JLD2OutputWriter(model, outputs,
                     schedule = TimeInterval(20minutes),
                     filename = "convecting_plankton.jld2",
                     overwrite_existing = true)

# !!! info "Using multiple output writers"
#     Because each output writer is associated with a single output `schedule`,
#     it often makes sense to use _different_ output writers for different types of output.
#     For example, smaller outputs that consume less disk space may be written more
#     frequently without threatening the capacity of your hard drive.
#     An arbitrary number of output writers may be added to `simulation.output_writers`.
#
# The simulation is set up. Let there be plankton:

run!(simulation)

# Notice how the time-step is reduced at early times, when turbulence is strong,
# and increases again towards the end of the simulation when turbulence fades.

# ## Visualizing the solution
#
# We'd like to a make a plankton movie. First we load the output file
# and build a time-series of the buoyancy flux,

using JLD2

filepath = simulation.output_writers[:simple_output].filepath

w_timeseries = FieldTimeSeries(filepath, "w")
P_timeseries = FieldTimeSeries(filepath, "P")
P_avg_timeseries = FieldTimeSeries(filepath, "P_avg")

times = w_timeseries.times
buoyancy_flux_time_series = [buoyancy_flux(0, 0, t, buoyancy_flux_parameters) for t in times]
nothing # hide

# and then we construct the ``x, z`` grid,

xw, yw, zw = nodes(w_timeseries)
xp, yp, zp = nodes(P_timeseries)
nothing # hide

# Finally, we animate plankton mixing and blooming,

using Plots

@info "Making a movie about plankton..."

w_lim = maximum(abs, interior(w_timeseries))

anim = @animate for i in 1:length(times)

    @info "Plotting frame $i of $(length(times))..."

    t = times[i]
    w = interior(w_timeseries[i], :, 1, :)
    P = interior(P_timeseries[i], :, 1, :)
    P_avg = interior(P_avg_timeseries[i], 1, 1, :)

    P_min = minimum(P) - 1e-9
    P_max = maximum(P) + 1e-9
    P_lims = (0.95, 1.1)

    w_levels = range(-w_lim, stop=w_lim, length=20)

    P_levels = collect(range(P_lims[1], stop=P_lims[2], length=20))
    P_lims[1] > P_min && pushfirst!(P_levels, P_min)
    P_lims[2] < P_max && push!(P_levels, P_max)

    kwargs = (xlabel="x (m)", ylabel="y (m)", aspectratio=1, linewidth=0, colorbar=true,
              xlims=(0, model.grid.Lx), ylims=(-model.grid.Lz, 0))

    w_contours = contourf(xw, zw, w';
                          color = :balance,
                          levels = w_levels,
                          clims = (-w_lim, w_lim),
                          kwargs...)

    P_contours = contourf(xp, zp, clamp.(P, P_lims[1], P_lims[2])';
                          color = :matter,
                          levels = P_levels,
                          clims = P_lims,
                          kwargs...)

    P_profile = plot(P_avg, zp,
                     linewidth = 2,
                     label = nothing,
                     xlims = (0.9, 1.3),
                     ylabel = "z (m)",
                     xlabel = "Plankton concentration (μM)")

    flux_plot = plot(times ./ hour, buoyancy_flux_time_series,
                     linewidth = 1,
                     label = "Buoyancy flux time series",
                     color = :black,
                     alpha = 0.4,
                     legend = :topright,
                     xlabel = "Time (hours)",
                     ylabel = "Buoyancy flux (m² s⁻³)",
                     ylims = (0.0, 1.1 * buoyancy_flux_parameters.initial_buoyancy_flux))

    plot!(flux_plot, times[1:i] ./ hour, buoyancy_flux_time_series[1:i],
          color = :steelblue,
          linewidth = 6,
          label = nothing)

    scatter!(flux_plot, times[i:i] / hour, buoyancy_flux_time_series[i:i],
             markershape = :circle,
             color = :steelblue,
             markerstrokewidth = 0,
             markersize = 15,
             label = "Current buoyancy flux")

    layout = Plots.grid(2, 2, widths=(0.7, 0.3))

    w_title = @sprintf("Vertical velocity (m s⁻¹) at %s", prettytime(t))
    P_title = @sprintf("Plankton concentration (μM) at %s", prettytime(t))

    plot(w_contours, flux_plot, P_contours, P_profile,
         title=[w_title "" P_title ""],
         layout=layout, size=(1000.5, 1000.5))
end

mp4(anim, "convecting_plankton.mp4", fps = 8) # hide
