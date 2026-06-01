# steel_casting_optimization
This repository contains the implementation of a "steel casting optimization problem" (Mixed Integer Linear Programming - MILP) migrated from CPU-based Gurobi to GPU-accelerated NVIDIA cuOpt.
## Overview

This repository contains the implementation of a **steel casting optimization problem** (Mixed Integer Linear Programming - MILP) migrated from CPU-based Gurobi to GPU-accelerated NVIDIA cuOpt. The project demonstrates how to:

- Build a complex MILP model for steel continuous casting
- Export the model to **MPS format** (universal solver format)
- Solve the same model using both Gurobi (CPU) and NVIDIA cuOpt (GPU)
- Compare performance metrics: solution time, optimality gap, and objective value

## Problem Description

The steel casting optimization problem involves scheduling coils across two parallel casters while minimizing:

- Grade change costs
- Gauge jump penalties
- Inter-caster width differences
- Trim losses
- Roller change costs

**Key constraints include:**
- Decreasing casting width requirements
- Roller degradation limits
- Heat tonnage restrictions
- Valid coil assignments

## Requirements

- Python 3.11 – 3.14
- Gurobi 11.0+ (with academic license)
- NVIDIA GPU (Volta architecture or newer)
- CUDA 12.0+ (for cuOpt)
- Operating System: Linux or Windows WSL2

## Acknowledgments

- Original authors: Torres et al. (2023) – Steel continuous casting optimization formulation
- NVIDIA cuOpt team for GPU-accelerated optimization engine
- Gurobi Optimization for academic licensing

## License

MIT License
