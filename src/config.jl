"""
    PathfinderConfig(; options...)

A configuration object for `Pathfinder.pathfinder`.

Wherever this object is used, `options` are passed as keyword arguments to `pathfinder`.
"""
struct PathfinderConfig{O<:NamedTuple}
    options::O
end
PathfinderConfig(; options...) = PathfinderConfig(NamedTuple(options))
