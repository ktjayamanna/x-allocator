# x-allocator

SmartConvert: Intelligent memory layout optimization for PyTorch models.

## Overview

x-allocator is a system that automatically optimizes `.contiguous()` placement in PyTorch models by profiling tensor layouts and strategically scheduling conversions during GPU idle time.

## Architecture

The system consists of three main components:

1. **PROFILER** - Profiles the model to detect non-contiguous tensors and measure conversion costs
2. **COST MODEL** - Learns a regression model to estimate conversion costs for unseen tensor shapes
3. **COMPILER** - Takes schedule.json and generates optimized PyTorch code with strategic `.contiguous()` placements

See `docs/sys.excalidraw` for the full system architecture diagram.

## Quick Start

### Generate Profiling Data

```bash
make profile
```

This will:
1. Run the profiler on your model
2. Train the cost model
3. Generate three JSON files in `data/tmp/`:

**profile.json** - Raw profiling data
- `records`: Detailed profiling data for each module
- `conversion_cost_table`: Measured `.contiguous()` costs
- `gpu_idle_events`: GPU idle time during data transfer with op references

**cost.json** - Cost model for debugging
- `conversion_cost_table`: Measured `.contiguous()` costs
- `cost_model`: Regression coefficients (α, β, γ)

**schedule.json** - Compiler input
- `ops`: Scheduler-ready operations list
- `gpu_idle_events`: GPU idle events with before_op_id and after_op_id

### Train Model

```bash
python src/train.py
```

## Project Structure

```
x-allocator/
├── src/
│   ├── profiler/          # Contiguity profiler
│   ├── cost_model/        # Cost estimation model
│   ├── compiler/          # Code generation (TODO)
│   ├── model.py           # MinimalGPT model
│   ├── train.py           # Training script
│   └── config.py          # Configuration
├── scripts/
│   └── profile.py         # Profile generation script
├── data/
│   └── tmp/               # Generated files (profile.json, etc.)
├── docs/                  # Architecture diagrams
└── Makefile               # Build targets
```

## Development

### Clean Generated Files

```bash
make clean
```

### Available Make Targets

- `make profile` - Generate profile.json
- `make clean` - Remove generated files
- `make help` - Show help message

## How It Works

### 1. Profiling Phase (Offline)

The profiler instruments the model with forward hooks to:
- Detect non-contiguous tensors at each layer
- Measure `.contiguous()` conversion costs
- Track GPU idle time during data transfer

### 2. Cost Model Phase (Offline)

The cost model learns a linear regression:
```
cost_ms ≈ α * numel + β * ndim + γ
```

This allows estimating conversion costs for unseen tensor shapes.

### 3. Compiler Phase (TODO)

The compiler will:
- Take profile.json as input
- Analyze the current PyTorch code
- Decide whether to call `.contiguous()` synchronously or asynchronously
- Place `.contiguous()` statements at optimal locations

## Configuration

Edit `src/config.py` to customize:
- Model architecture (embedding size, layers, heads)
- Training parameters (batch size, learning rate, epochs)
- Device (CPU/GPU)

