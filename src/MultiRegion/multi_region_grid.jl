using Oceananigans.Grids: metrics_precomputed, on_architecture, pop_flat_elements
import Oceananigans.Grids: architecture, size, new_data, halo_size

struct MultiRegionGrid{FT, TX, TY, TZ, P, G, D, Arch} <: AbstractMultiRegionGrid{FT, TX, TY, TZ, Arch}
    architecture :: Arch
    partition :: P
    region_grids :: G
    devices :: D

    function MultiRegionGrid{FT, TX, TY, TZ}(arch::A, partition::P,
                                             region_grids::G,
                                             devices::D) where {FT, TX, TY, TZ, P, G, D, A}

        return new{FT, TX, TY, TZ, P, G, D, A}(arch, partition, region_grids, devices)
    end
end

isregional(mrg::MultiRegionGrid)        = true
getdevice(mrg::MultiRegionGrid, i)      = getdevice(mrg.region_grids, i)
switch_device!(mrg::MultiRegionGrid, i) = switch_device!(getdevice(mrg, i))
devices(mrg::MultiRegionGrid)           = devices(mrg.region_grids)
sync_all_devices!(mrg::MultiRegionGrid) = sync_all_devices!(devices(mrg))

getregion(mrg::MultiRegionGrid, i)  = getregion(mrg.region_grids, i)
Base.length(mrg::MultiRegionGrid)   = Base.length(mrg.region_grids)

const ImmersedMultiRegionGrid = MultiRegionGrid{FT, TX, TY, TZ, P, <:MultiRegionObject{<:Tuple{Vararg{<:ImmersedBoundaryGrid}}}} where {FT, TX, TY, TZ, P}

function MultiRegionGrid(global_grid; partition = XPartition(2), devices = nothing)

    if length(partition) == 1
        return global_grid
    end

    @warn "MultiRegion functionalities are experimental: help the development by reporting bugs or non-implemented features!"

    arch    = devices isa Nothing ? CPU() : GPU()
    devices = validate_devices(partition, devices)
    devices = assign_devices(partition, devices)

    global_grid  = on_architecture(CPU(), global_grid)
    local_size   = MultiRegionObject(partition_size(partition, global_grid), devices)
    local_extent = MultiRegionObject(partition_extent(partition, global_grid), devices)
    local_topo   = MultiRegionObject(partition_topology(partition, global_grid), devices)  
    
    global_topo  = topology(global_grid)

    FT   = eltype(global_grid)
    
    args = (Reference(global_grid), 
            Reference(arch), 
            local_topo, 
            local_size,
            local_extent, 
            Reference(partition), 
            Iterate(1:length(partition)))

    region_grids = construct_regionally(construct_grid, args...)

    return MultiRegionGrid{FT, global_topo[1], global_topo[2], global_topo[3]}(arch, partition, region_grids, devices)
end

function construct_grid(grid::RectilinearGrid, child_arch, topo, size, extent, args...)
    halo = halo_size(grid)
    size = pop_flat_elements(size, topo)
    halo = pop_flat_elements(halo, topo)
    FT   = eltype(grid)
    return RectilinearGrid(child_arch, FT; size = size, halo = halo, topology = topo, extent...)
end

function construct_grid(grid::LatitudeLongitudeGrid, child_arch, topo, size, extent, args...)
    halo = halo_size(grid)
    FT   = eltype(grid)
    lon, lat, z = extent
    return LatitudeLongitudeGrid(child_arch, FT; 
                                 size = size, halo = halo, radius = grid.radius,
                                 latitude = lat, longitude = lon, z = z, topology = topo,
                                 precompute_metrics = metrics_precomputed(grid))
end

function construct_grid(ibg::ImmersedBoundaryGrid, child_arch, topo, local_size, extent, partition, region)
    boundary = partition_immersed_boundary(ibg.immersed_boundary, partition, local_size, region, child_arch)
    return ImmersedBoundaryGrid(construct_grid(ibg.grid, child_arch, topo, local_size, extent), boundary)
end

partition_immersed_boundary(b, args...) = 
    getname(b)(partition_global_array(getproperty(b, propertynames(b)[1]), args...))

function reconstruct_global_grid(mrg)
    size    = reconstruct_size(mrg, mrg.partition)
    extent  = reconstruct_extent(mrg, mrg.partition)
    topo    = topology(mrg)
    switch_device!(mrg.devices[1])
    return construct_grid(mrg.region_grids[1], architecture(mrg), topo, size, extent)
end

function reconstruct_global_grid(mrg::ImmersedMultiRegionGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    underlying_mrg = MultiRegionGrid{FT, TX, TY, TZ}(architecture(mrg), 
                                                     mrg.partition, 
                                                     construct_regionally(getproperty, mrg, :grid), 
                                                     mrg.devices)
                                                     
    global_grid     = on_architecture(CPU(), reconstruct_global_grid(underlying_mrg))
    cpu_mrg         = on_architecture(CPU(), mrg)
    local_boundary  = construct_regionally(getproperty, cpu_mrg, :immersed_boundary)
    local_array     = construct_regionally(getproperty, local_boundary, propertynames(local_boundary[1])[1])
    local_array     = construct_regionally(getinterior, local_array, mrg)
    global_boundary = getname(local_boundary[1])(reconstruct_global_array(local_array, mrg.partition, architecture(mrg)))
    return on_architecture(architecture(mrg), ImmersedBoundaryGrid(global_grid, global_boundary))
end

getinterior(array::AbstractArray{T, 2}, grid) where T = array[1:grid.Nx, 1:grid.Ny]
getinterior(array::AbstractArray{T, 3}, grid) where T = array[1:grid.Nx, 1:grid.Ny, 1:grid.Nz]
getinterior(func::Function, grid) = func

multi_region_object_from_array(a::AbstractArray, grid) = arch_array(architecture(grid), a)

function multi_region_object_from_array(a::AbstractArray, mrg::MultiRegionGrid)
    local_size = construct_regionally(size, mrg)
    arch = architecture(mrg)
    a    = arch_array(CPU(), a)
    ma   = construct_regionally(partition_global_array, a, mrg.partition, local_size, Iterate(1:length(mrg)), arch)
    return ma
end

import Oceananigans.Grids: with_halo, on_architecture

function with_halo(new_halo, mrg::MultiRegionGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    new_grids = construct_regionally(with_halo, new_halo, mrg)
    return MultiRegionGrid{FT, TX, TY, TZ}(mrg.architecture, mrg.partition, new_grids, mrg.devices)
end

function on_architecture(::CPU, mrg::MultiRegionGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    new_grids = construct_regionally(on_architecture, CPU(), mrg)
    devices   = Tuple(CPU() for i in 1:length(mrg))  
    return MultiRegionGrid{FT, TX, TY, TZ}(CPU(), mrg.partition, new_grids, devices)
end

Base.summary(mrg::MultiRegionGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =  
    "MultiRegionGrid{$FT, $TX, $TY, $TZ} with $(summary(mrg.partition)) on $(string(typeof(mrg.region_grids[1]).name.wrapper))"

Base.show(io::IO, mrg::MultiRegionGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =  
    print(io, "MultiRegionGrid{$FT, $TX, $TY, $TZ} partitioned on $(architecture(mrg)): \n",
              "├── grids: $(summary(mrg.region_grids[1])) \n",
              "├── partitioning: $(summary(mrg.partition)) \n",
              "└── devices: $(devices(mrg))")

#### 
#### Utilitites that aren't used anywhere at the moment
####

grids(mrg::MultiRegionGrid) = mrg.region_grids

getmultiproperty(mrg::MultiRegionGrid, x::Symbol) = construct_regionally(Base.getproperty, grids(mrg), x)

const MRG = MultiRegionGrid

@inline Base.getproperty(ibg::MRG, property::Symbol) = get_multi_property(ibg, Val(property))
@inline get_multi_property(ibg::MRG, ::Val{property}) where property = getproperty(getindex(getfield(ibg, :region_grids), 1), property)
@inline get_multi_property(ibg::MRG, ::Val{:partition}) = getfield(ibg, :partition)
@inline get_multi_property(ibg::MRG, ::Val{:region_grids}) = getfield(ibg, :region_grids)
@inline get_multi_property(ibg::MRG, ::Val{:devices}) = getfield(ibg, :devices)
