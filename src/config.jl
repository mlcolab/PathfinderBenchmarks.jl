struct PathfinderConfig{O}
    optimizer::O
end
function PathfinderConfig()
    return PathfinderConfig(Pathfinder.default_optimizer(Pathfinder.DEFAULT_HISTORY_LENGTH))
end
