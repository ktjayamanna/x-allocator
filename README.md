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
- `records`: Detailed profiling data for each module with tensor IDs
- `conversion_cost_table`: Measured `.contiguous()` costs
- `gpu_idle_events`: GPU idle time during data transfer with op references
- `tensor_flow`: Tensor-level data flow graph (producer-consumer relationships)

**cost.json** - Cost model for debugging
- `conversion_cost_table`: Measured `.contiguous()` costs
- `cost_model`: Regression coefficients (α, β, γ)

**schedule.json** - Compiler input
- `ops`: Scheduler-ready operations list with tensor IDs
- `gpu_idle_events`: GPU idle events with before_op_id and after_op_id
- `tensor_flow`: Tensor-level data flow graph for optimization

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
- **Build tensor-level data flow graph** (producer-consumer relationships)

Key innovation: **Tensor-level tracking** instead of module-level tracking
- Tracks which tensors are shared across multiple operations
- Identifies optimization opportunities (e.g., convert once at producer instead of N times at consumers)

### 2. Cost Model Phase (Offline)

The cost model learns a linear regression:
```
cost_ms ≈ α * numel + β * ndim + γ
```

This allows estimating conversion costs for unseen tensor shapes.

### 3. Compiler Phase (TODO)

The compiler will:
- Take schedule.json as input (includes tensor flow graph)
- Analyze the current PyTorch code
- **Use tensor flow graph to identify shared tensors**
- **Place `.contiguous()` at producers to avoid redundant conversions**
- Schedule `.contiguous()` during GPU idle times
- Insert `.contiguous()` statements at optimal source code locations

## Tensor-Level Tracking

### Why Tensor-Level Instead of Module-Level?

**Module-level tracking** (traditional approach):
- Tracks which modules have non-contiguous inputs/outputs
- Misses optimization opportunities when the same tensor is used by multiple modules
- May insert redundant `.contiguous()` calls

**Tensor-level tracking** (our approach):
- Tracks producer-consumer relationships for each tensor
- Identifies shared tensors (same tensor consumed by multiple ops)
- Enables smarter `.contiguous()` placement

### Example Optimization

Consider this code:
```python
y = layer1(x)  # y is non-contiguous
z1 = layer2(y)  # Uses y
z2 = layer3(y)  # Uses same y
```

**Without tensor tracking:**
```python
z1 = layer2(y.contiguous())  # Convert
z2 = layer3(y.contiguous())  # Convert again (redundant!)
```

**With tensor tracking:**
```python
y = layer1(x).contiguous()  # Convert once at producer
z1 = layer2(y)  # Reuse
z2 = layer3(y)  # Reuse
```

The compiler can make this optimization because it knows:
- Tensor `y` is produced by `layer1` (op_0)
- Tensor `y` is consumed by `layer2` (op_1) and `layer3` (op_2)
- Therefore, convert once at the producer instead of twice at consumers

## Configuration

Edit `src/config.py` to customize:
- Model architecture (embedding size, layers, heads)
- Training parameters (batch size, learning rate, epochs)
- Device (CPU/GPU)

