struct PathfinderConfig{O<:NamedTuple}
    options::O
end
PathfinderConfig(; options...) = PathfinderConfig(NamedTuple(options))
