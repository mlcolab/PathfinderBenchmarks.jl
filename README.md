# PathfinderBenchmarks

[![Build Status](https://github.com/mlcolab/PathfinderBenchmarks.jl/workflows/CI/badge.svg)](https://github.com/mlcolab/PathfinderBenchmarks.jl/actions)
[![Coverage](https://codecov.io/gh/mlcolab/PathfinderBenchmarks.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mlcolab/PathfinderBenchmarks.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

Benchmark code and results for [Pathfinder.jl](https://github.com/mlcolab/Pathfinder.jl).

# Instructions

All analyses are performed in reproducible Pluto notebooks in `notebooks`.
While they may be run interactively in Pluto, they are also Julia scripts.
To perform the analyses, run

```bash
julia notebooks/choose_hmc_config.jl  # can be skipped, as the results are committed
julia notebooks/run_hmc_benchmark.jl
julia notebooks/run_pathfinder_benchmark.jl
julia notebooks/summarize_benchmark_results.jl
julia notebooks/plot_benchmark_results.jl
julia notebooks/plot_illustrations.jl
```
