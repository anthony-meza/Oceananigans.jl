using CUDA: has_cuda
using OrderedCollections: OrderedDict

using Oceananigans: AbstractModel, AbstractOutputWriter, AbstractDiagnostic

using Oceananigans.Architectures: AbstractArchitecture, GPU
using Oceananigans.Advection: AbstractAdvectionScheme, CenteredSecondOrder, VectorInvariantSchemes, VectorInvariant, WENOVectorInvariant
using Oceananigans.BuoyancyModels: validate_buoyancy, regularize_buoyancy, SeawaterBuoyancy, g_Earth
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Fields: Field, CenterField, tracernames, VelocityFields, TracerFields
using Oceananigans.Forcings: model_forcing
using Oceananigans.Grids: halo_size, inflate_halo_size, with_halo, AbstractRectilinearGrid
using Oceananigans.Grids: AbstractCurvilinearGrid, AbstractHorizontallyCurvilinearGrid, architecture
using Oceananigans.Models.NonhydrostaticModels: extract_boundary_conditions
using Oceananigans.TimeSteppers: Clock, TimeStepper, update_state!
using Oceananigans.TurbulenceClosures: validate_closure, with_tracers, DiffusivityFields, add_closure_specific_boundary_conditions
using Oceananigans.TurbulenceClosures: time_discretization, implicit_diffusion_solver
using Oceananigans.LagrangianParticleTracking: LagrangianParticles
using Oceananigans.Utils: tupleit

""" Returns a default_tracer_advection, tracer_advection `tuple`. """
validate_tracer_advection(invalid_tracer_advection, grid) = error("$invalid_tracer_advection is invalid tracer_advection!")
validate_tracer_advection(tracer_advection_tuple::NamedTuple, grid) = CenteredSecondOrder(), tracer_advection_tuple
validate_tracer_advection(tracer_advection::AbstractAdvectionScheme, grid) = tracer_advection, NamedTuple()

PressureField(grid) = (; pHY′ = CenterField(grid))

mutable struct HydrostaticFreeSurfaceModel{TS, E, A<:AbstractArchitecture, S,
                                           G, T, V, B, R, F, P, U, C, Φ, K, AF} <: AbstractModel{TS}
  
          architecture :: A        # Computer `Architecture` on which `Model` is run
                  grid :: G        # Grid of physical points on which `Model` is solved
                 clock :: Clock{T} # Tracks iteration number and simulation time of `Model`
             advection :: V        # Advection scheme for tracers
              buoyancy :: B        # Set of parameters for buoyancy model
              coriolis :: R        # Set of parameters for the background rotation rate of `Model`
          free_surface :: S        # Free surface parameters and fields
               forcing :: F        # Container for forcing functions defined by the user
               closure :: E        # Diffusive 'turbulence closure' for all model fields
             particles :: P        # Particle set for Lagrangian tracking
            velocities :: U        # Container for velocity fields `u`, `v`, and `w`
               tracers :: C        # Container for tracer fields
              pressure :: Φ        # Container for hydrostatic pressure
    diffusivity_fields :: K        # Container for turbulent diffusivities
           timestepper :: TS       # Object containing timestepper fields and parameters
      auxiliary_fields :: AF       # User-specified auxiliary fields for forcing functions and boundary conditions
end

"""
    HydrostaticFreeSurfaceModel(; grid,
                                             clock = Clock{eltype(grid)}(0, 0, 1),
                                momentum_advection = CenteredSecondOrder(),
                                  tracer_advection = CenteredSecondOrder(),
                                          buoyancy = SeawaterBuoyancy(eltype(grid)),
                                          coriolis = nothing,
                                      free_surface = ExplicitFreeSurface(gravitational_acceleration=g_Earth),
                               forcing::NamedTuple = NamedTuple(),
                                           closure = nothing,
                   boundary_conditions::NamedTuple = NamedTuple(),
                                           tracers = (:T, :S),
    particles::Union{Nothing, LagrangianParticles} = nothing,
                                        velocities = nothing,
                                          pressure = nothing,
                                diffusivity_fields = nothing,
                                  auxiliary_fields = NamedTuple(),
    )

Construct a hydrostatic model with a free surface on `grid`.

Keyword arguments
=================

  - `grid`: (required) The resolution and discrete geometry on which `model` is solved. The
    architecture (CPU/GPU) that the model is solve is inferred from the architecture of the grid.
  - `momentum_advection`: The scheme that advects velocities. See `Oceananigans.Advection`.
  - `tracer_advection`: The scheme that advects tracers. See `Oceananigans.Advection`.
  - `buoyancy`: The buoyancy model. See `Oceananigans.BuoyancyModels`.
  - `coriolis`: Parameters for the background rotation rate of the model.
  - `forcing`: `NamedTuple` of user-defined forcing functions that contribute to solution tendencies.
  - `free_surface`: The free surface model.
  - `closure`: The turbulence closure for `model`. See `Oceananigans.TurbulenceClosures`.
  - `boundary_conditions`: `NamedTuple` containing field boundary conditions.
  - `tracers`: A tuple of symbols defining the names of the modeled tracers, or a `NamedTuple` of
               preallocated `CenterField`s.
  - `particles`: Lagrangian particles to be advected with the flow. Default: `nothing`.
  - `velocities`: The model velocities. Default: `nothing`.
  - `pressure`: Hydrostatic pressure field. Default: `nothing`.
  - `diffusivity_fields`: Diffusivity fields. Default: `nothing`.
  - `auxiliary_fields`: `NamedTuple` of auxiliary fields. Default: `nothing`.

"""
function HydrostaticFreeSurfaceModel(; grid,
                                             clock = Clock{eltype(grid)}(0, 0, 1),
                                momentum_advection = CenteredSecondOrder(),
                                  tracer_advection = CenteredSecondOrder(),
                                          buoyancy = SeawaterBuoyancy(eltype(grid)),
                                          coriolis = nothing,
                                      free_surface = ExplicitFreeSurface(gravitational_acceleration=g_Earth),
                               forcing::NamedTuple = NamedTuple(),
                                           closure = nothing,
                   boundary_conditions::NamedTuple = NamedTuple(),
                                           tracers = (:T, :S),
    particles::Union{Nothing, LagrangianParticles} = nothing,
                                        velocities = nothing,
                                          pressure = nothing,
                                diffusivity_fields = nothing,
                                  auxiliary_fields = NamedTuple(),
    )

    # Check halos and throw an error if the grid's halo is too small
    user_halo = halo_size(grid)
    required_halo = inflate_halo_size(user_halo..., topology(grid),
                                      momentum_advection,
                                      tracer_advection,
                                      closure)

    any(user_halo .< required_halo) &&
        throw(ArgumentError("The grid halo $user_halo must be larger than $required_halo."))

    arch = architecture(grid)

    arch == GPU() && !has_cuda() &&
         throw(ArgumentError("Cannot create a GPU model. No CUDA-enabled GPU was detected!"))

    momentum_advection = validate_momentum_advection(momentum_advection, grid)

    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)

    validate_buoyancy(buoyancy, tracernames(tracers))
    buoyancy = regularize_buoyancy(buoyancy)

    # Collect boundary conditions for all model prognostic fields and, if specified, some model
    # auxiliary fields. Boundary conditions are "regularized" based on the _name_ of the field:
    # boundary conditions on u, v are regularized assuming they represent momentum at appropriate
    # staggered locations. All other fields are regularized assuming they are tracers.
    # Note that we do not regularize boundary conditions contained in *tupled* diffusivity fields right now.
    #
    # First, we extract boundary conditions that are embedded within any _user-specified_ field tuples:
    embedded_boundary_conditions = merge(extract_boundary_conditions(velocities),
                                         extract_boundary_conditions(tracers),
                                         extract_boundary_conditions(pressure),
                                         extract_boundary_conditions(diffusivity_fields))

    # Next, we form a list of default boundary conditions:
    prognostic_field_names = (:u, :v, :η, tracernames(tracers)...)
    default_boundary_conditions = NamedTuple{prognostic_field_names}(Tuple(FieldBoundaryConditions() for name in prognostic_field_names))

    # Then we merge specified, embedded, and default boundary conditions. Specified boundary conditions
    # have precedence, followed by embedded, followed by default.
    boundary_conditions = merge(default_boundary_conditions, embedded_boundary_conditions, boundary_conditions)
    boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, prognostic_field_names)

    # Finally, we ensure that closure-specific boundary conditions, such as
    # those required by TKEBasedVerticalDiffusivity, are enforced:
    boundary_conditions = add_closure_specific_boundary_conditions(closure, boundary_conditions, grid, tracernames(tracers), buoyancy)

    # Ensure `closure` describes all tracers
    closure = with_tracers(tracernames(tracers), closure)

    # Put CATKE first in the list of closures
    closure = validate_closure(closure)

    # Either check grid-correctness, or construct tuples of fields
    velocities         = HydrostaticFreeSurfaceVelocityFields(velocities, grid, clock, boundary_conditions)
    tracers            = TracerFields(tracers, grid, boundary_conditions)
    pressure           = PressureField(grid)
    diffusivity_fields = DiffusivityFields(diffusivity_fields, grid, tracernames(tracers), boundary_conditions, closure)

    validate_velocity_boundary_conditions(velocities)

    free_surface = FreeSurface(free_surface, velocities, grid)

    # Instantiate timestepper if not already instantiated
    implicit_solver = implicit_diffusion_solver(time_discretization(closure), grid)
    timestepper = TimeStepper(:QuasiAdamsBashforth2, grid, tracernames(tracers);
                              implicit_solver = implicit_solver,
                              Gⁿ = HydrostaticFreeSurfaceTendencyFields(velocities, free_surface, grid, tracernames(tracers)),
                              G⁻ = HydrostaticFreeSurfaceTendencyFields(velocities, free_surface, grid, tracernames(tracers)))

    # Regularize forcing for model tracer and velocity fields.
    model_fields = hydrostatic_prognostic_fields(velocities, free_surface, tracers)
    forcing = model_forcing(model_fields; forcing...)

    default_tracer_advection, tracer_advection = validate_tracer_advection(tracer_advection, grid)

    # Advection schemes
    tracer_advection_tuple = with_tracers(tracernames(tracers),
                                          tracer_advection,
                                          (name, tracer_advection) -> default_tracer_advection,
                                          with_velocities=false)

    advection = merge((momentum=momentum_advection,), tracer_advection_tuple)

    model = HydrostaticFreeSurfaceModel(arch, grid, clock, advection, buoyancy, coriolis,
                                        free_surface, forcing, closure, particles, velocities, tracers,
                                        pressure, diffusivity_fields, timestepper, auxiliary_fields)

    update_state!(model)

    return model
end

validate_velocity_boundary_conditions(velocities) = validate_vertical_velocity_boundary_conditions(velocities.w)

function validate_vertical_velocity_boundary_conditions(w)
    w.boundary_conditions.top === nothing || error("Top boundary condition for HydrostaticFreeSurfaceModel velocities.w
                                                    must be `nothing`!")
    return nothing
end

momentum_advection_squawk(momentum_advection, grid) = error("$(typeof(momentum_advection)) is not supported with $(typeof(grid))")

validate_momentum_advection(momentum_advection, grid) = momentum_advection
validate_momentum_advection(momentum_advection, grid::AbstractHorizontallyCurvilinearGrid) = momentum_advection_squawk(momentum_advection, grid)
validate_momentum_advection(momentum_advection::Union{VectorInvariantSchemes, Nothing}, grid::AbstractHorizontallyCurvilinearGrid) = momentum_advection
