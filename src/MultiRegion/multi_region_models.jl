using Oceananigans.Models: AbstractModel
using Oceananigans.Advection: WENO5
using Oceananigans.Models.HydrostaticFreeSurfaceModels: AbstractFreeSurface
using Oceananigans.TimeSteppers: AbstractTimeStepper, QuasiAdamsBashforth2TimeStepper
using Oceananigans.Models: PrescribedVelocityFields
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.Advection: AbstractAdvectionScheme

import Oceananigans.Simulations: new_time_step
import Oceananigans.Diagnostics: accurate_cell_advection_timescale
import Oceananigans.Advection: WENO5
import Oceananigans.Models.HydrostaticFreeSurfaceModels: build_implicit_step_solver, validate_tracer_advection
import Oceananigans.TurbulenceClosures: implicit_diffusion_solver

const MultiRegionModel = HydrostaticFreeSurfaceModel{<:Any, <:Any, <:AbstractArchitecture, <:Any, <:MultiRegionGrid}

# Bottleneck is getregion!!! (there are type issues with FieldBoundaryConditions and with propertynames)
for T in (:HydrostaticFreeSurfaceModel, :QuasiAdamsBashforth2TimeStepper, :PrescribedVelocityFields)
    @eval begin
        # This assumes a constructor of the form T(arg1, arg2, ...) exists,
        # which is not the case for all types.
        @inline getregion(t::$T, r) = $T(Tuple(getregion(getproperty(t, n), r) for n in fieldnames($T))...)
    end
end

@inline getregion(fs::ExplicitFreeSurface, r) = ExplicitFreeSurface(getregion(fs.η, r), fs.gravitational_acceleration)
@inline isregional(pv::PrescribedVelocityFields) = isregional(pv.u) | isregional(pv.v) | isregional(pv.w)
@inline devices(pv::PrescribedVelocityFields)    = devices(pv[findfirst(isregional, (pv.u, pv.v, pv.w))])

validate_tracer_advection(tracer_advection::MultiRegionObject, grid::MultiRegionGrid) = tracer_advection, NamedTuple()

@inline isregional(mrm::MultiRegionModel)        = true
@inline devices(mrm::MultiRegionModel)           = devices(mrm.grid)
@inline getdevice(mrm::MultiRegionModel, d)      = getdevice(mrm.grid, d)
@inline switch_region!(mrm::MultiRegionModel, d) = switch_region!(mrm.grid, d)

implicit_diffusion_solver(time_discretization::VerticallyImplicitTimeDiscretization, mrg::MultiRegionGrid) =
      construct_regionally(implicit_diffusion_solver, time_discretization, mrg)

WENO5(mrg::MultiRegionGrid, args...; kwargs...) = construct_regionally(WENO5, mrg, args...; kwargs...)

function accurate_cell_advection_timescale(grid::MultiRegionGrid, velocities)
    Δt = construct_regionally(accurate_cell_advection_timescale, grid, velocities)
    return minimum(Δt.regions)
end

function new_time_step(old_Δt, wizard, model::MultiRegionModel)
    Δt = construct_regionally(new_time_step, old_Δt, wizard, model)
    return minimum(Δt.regions)
end
