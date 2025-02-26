using Glob
import Oceananigans.Fields: set!

using Oceananigans: fields
using Oceananigans.Fields: offset_data
using Oceananigans.TimeSteppers: RungeKutta3TimeStepper, QuasiAdamsBashforth2TimeStepper

mutable struct Checkpointer{T, P} <: AbstractOutputWriter
    schedule :: T
    dir :: String
    prefix :: String
    properties :: P
    overwrite_existing :: Bool
    verbose :: Bool
    cleanup :: Bool
end

"""
    Checkpointer(model; schedule,
                 dir = ".",
                 prefix = "checkpoint",
                 overwrite_existing = false,
                 verbose = false,
                 cleanup = false,
                 properties = [:architecture, :grid, :clock, :coriolis,
                               :buoyancy, :closure, :velocities, :tracers,
                               :timestepper, :particles]
                )

Construct a `Checkpointer` that checkpoints the model to a JLD2 file on `schedule.`
The `model.clock.iteration` is included in the filename to distinguish between multiple checkpoint files.

To restart or "pickup" a model from a checkpoint, specify `pickup=true` when calling `run!`, ensuring
that the checkpoint file is the current working directory. See 

```julia
help> run!
```
for more details.

Note that extra model `properties` can be safely specified, but removing crucial properties
such as `:velocities` will make restoring from the checkpoint impossible.

The checkpointer attempts to serialize as much of the model to disk as possible,
but functions or objects containing functions cannot be serialized at this time.

Keyword arguments
=================
- `schedule` (required): Schedule that determines when to checkpoint.

- `dir`: Directory to save output to. Default: "." (current working directory).

- `prefix`: Descriptive filename prefixed to all output files. Default: "checkpoint".

- `overwrite_existing`: Remove existing files if their filenames conflict. Default: `false`.

- `verbose`: Log what the output writer is doing with statistics on compute/write times
             and file sizes. Default: `false`.

- `cleanup`: Previous checkpoint files will be deleted once a new checkpoint file is written.
             Default: `false`.

- `properties`: List of model properties to checkpoint. Some are required.
"""
function Checkpointer(model; schedule,
                      dir = ".",
                      prefix = "checkpoint",
                      overwrite_existing = false,
                      verbose = false,
                      cleanup = false,
                      properties = [:architecture, :grid, :clock, :coriolis,
                                    :buoyancy, :closure, :timestepper, :particles])

    # Certain properties are required for `restore_from_checkpoint` to work.
    required_properties = (:grid, :architecture, :timestepper, :particles)

    for rp in required_properties
        if rp ∉ properties
            @warn "$rp is required for checkpointing. It will be added to checkpointed properties"
            push!(properties, rp)
        end
    end

    for p in properties
        p isa Symbol || error("Property $p to be checkpointed must be a Symbol.")
        p ∉ propertynames(model) && error("Cannot checkpoint $p, it is not a model property!")

        if (p ∉ required_properties) && has_reference(Function, getproperty(model, p))
            @warn "model.$p contains a function somewhere in its hierarchy and will not be checkpointed."
            filter!(e -> e != p, properties)
        end
    end

    mkpath(dir)

    return Checkpointer(schedule, dir, prefix, properties, overwrite_existing, verbose, cleanup)
end

#####
##### Checkpointer utils
#####

""" Returns the full prefix (the `superprefix`) associated with `checkpointer`. """
checkpoint_superprefix(prefix) = prefix * "_iteration"

"""
    checkpoint_path(iteration::Int, c::Checkpointer)

Returns the path to the `c`heckpointer file associated with model `iteration`.
"""
checkpoint_path(iteration::Int, c::Checkpointer) =
    joinpath(c.dir, string(checkpoint_superprefix(c.prefix), iteration, ".jld2"))

# This is the default name used in the simulation.output_writers ordered dict.
defaultname(::Checkpointer, nelems) = :checkpointer

""" Returns `filepath`. Shortcut for `run!(simulation, pickup=filepath)`. """
checkpoint_path(filepath::AbstractString, output_writers) = filepath

function checkpoint_path(pickup, output_writers)
    checkpointers = filter(writer -> writer isa Checkpointer, collect(values(output_writers)))
    length(checkpointers) == 0 && error("No checkpointers found: cannot pickup simulation!")
    length(checkpointers) > 1 && error("Multiple checkpointers found: not sure which one to pickup simulation from!")
    return checkpoint_path(pickup, first(checkpointers))
end

"""
    checkpoint_path(pickup::Bool, checkpointer)

For `pickup=true`, parse the filenames in `checkpointer.dir` associated with
`checkpointer.prefix` and return the path to the file whose name contains
the largest iteration.
"""
function checkpoint_path(pickup::Bool, checkpointer::Checkpointer)
    filepaths = glob(checkpoint_superprefix(checkpointer.prefix) * "*.jld2", checkpointer.dir)

    if length(filepaths) == 0 # no checkpoint files found
        # https://github.com/CliMA/Oceananigans.jl/issues/1159
        @warn "pickup=true but no checkpoints were found. Simulation will run without picking up."
        return nothing
    else
        return latest_checkpoint(checkpointer, filepaths)
    end
end

function latest_checkpoint(checkpointer, filepaths)
    filenames = basename.(filepaths)
    leading = length(checkpoint_superprefix(checkpointer.prefix))
    trailing = length(".jld2") # 5
    iterations = map(name -> parse(Int, name[leading+1:end-trailing]), filenames)
    latest_iteration, idx = findmax(iterations)
    return filepaths[idx]
end

#####
##### Writing checkpoints
#####

function write_output!(c::Checkpointer, model)
    filepath = checkpoint_path(model.clock.iteration, c)
    c.verbose && @info "Checkpointing to file $filepath..."

    t1 = time_ns()
    jldopen(filepath, "w") do file
        file["checkpointed_properties"] = c.properties
        serializeproperties!(file, model, c.properties)

        model_fields = fields(model)
        field_names = keys(model_fields)
        for name in field_names
            serializeproperty!(file, string(name), model_fields[name])
        end
    end

    t2, sz = time_ns(), filesize(filepath)
    c.verbose && @info "Checkpointing done: time=$(prettytime((t2-t1)/1e9)), size=$(pretty_filesize(sz))"

    c.cleanup && cleanup_checkpoints(c)

    return nothing
end

function cleanup_checkpoints(checkpointer)
    filepaths = glob(checkpoint_superprefix(checkpointer.prefix) * "*.jld2", checkpointer.dir)
    latest_checkpoint_filepath = latest_checkpoint(checkpointer, filepaths)
    [rm(filepath) for filepath in filepaths if filepath != latest_checkpoint_filepath]
    return nothing
end

#####
##### set! for checkpointer filepaths
#####

set!(model, ::Nothing) = nothing

"""
    set!(model, filepath::AbstractString)

Set data in `model.velocities`, `model.tracers`, `model.timestepper.Gⁿ`, and
`model.timestepper.G⁻` to checkpointed data stored at `filepath`.
"""
function set!(model, filepath::AbstractString)

    jldopen(filepath, "r") do file

        # Validate the grid
        checkpointed_grid = file["grid"]

	if model.grid isa ImmersedBoundaryGrid
            model.grid.grid == checkpointed_grid || 
            error("The grid associated with $filepath and the underlying `ImmersedBoundaryGrid.grid` are not the same!")
        else
         model.grid == checkpointed_grid ||
             error("The grid associated with $filepath and model.grid are not the same!")
	end

        model_fields = fields(model)

        for name in propertynames(model_fields)
            try
                parent_data = file["$name/data"]
                model_field = model_fields[name]
                copyto!(model_field.data.parent, parent_data)
            catch
                @warn "Could not retore $name from checkpoint."
            end
        end

        set_time_stepper!(model.timestepper, file, model_fields)

        if !isnothing(model.particles)
            copyto!(model.particles.properties, file["particles"])
        end

        checkpointed_clock = file["clock"]

        # Update model clock
        model.clock.iteration = checkpointed_clock.iteration
        model.clock.time = checkpointed_clock.time
    end

    return nothing
end

function set_time_stepper_tendencies!(timestepper, file, model_fields)
    for name in propertynames(model_fields)
        # Tendency "n"
        parent_data = file["timestepper/Gⁿ/$name/data"]

        tendencyⁿ_field = timestepper.Gⁿ[name]
        copyto!(tendencyⁿ_field.data.parent, parent_data)

        # Tendency "n-1"
        parent_data = file["timestepper/G⁻/$name/data"]

        tendency⁻_field = timestepper.G⁻[name]
        copyto!(tendency⁻_field.data.parent, parent_data)
    end

    return nothing
end

set_time_stepper!(timestepper::RungeKutta3TimeStepper, file, model_fields) =
    set_time_stepper_tendencies!(timestepper, file, model_fields)

function set_time_stepper!(timestepper::QuasiAdamsBashforth2TimeStepper, file, model_fields)
    set_time_stepper_tendencies!(timestepper, file, model_fields)
    timestepper.previous_Δt = file["timestepper/previous_Δt"]
    return nothing
end
            
