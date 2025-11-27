# Minimal Runtime Scheduler (For a PyTorch Training Loop)

## Overview

This project provides a compact reference implementation of a runtime scheduler that integrates with a simplified PyTorch-like training flow. Its purpose is to make the control points, execution order, and parameter-lifecycle management of a scheduler **explicit and inspectable**, without relying on PyTorch internals.

The components include:

* A toy “model” composed of simple layers with a `forward(x)` interface
* A training loop with forward and backward traversal
* A scheduler object that receives hook callbacks from the training loop
* A schedule representation describing when parameter operations occur
* Optional support for prefetching behavior

The result is a minimal working example of how runtime schedulers orchestrate fetch, compute, and release operations around model execution.

---

## High-Level Structure

The simplified stack looks like:

* **Fake layers**: each with a `.forward(x)` method.
* **FakeModel**: sequentially executes layers.
* **Training loop**: emulates a PyTorch-like step over batches.
* **Scheduler**: reacts to hook calls from the training loop via:

  * `on_step_start()`
  * `on_pre_forward(layer)`
  * `on_post_forward(layer)`
  * `on_pre_backward(layer)`
  * `on_post_backward(layer)`
  * `on_step_end()`

Internally, the scheduler simulates parameter-lifecycle operations:

* `fetch(layer)`   — bring layer parameters “on device”
* `release(layer)` — offload parameters
* `compute(layer)` — conceptually run the layer computation

A **schedule** is simply an ordered list of these operations, independent of wall-clock timing.

---

## Project Layout

```
minimal_scheduler/
  __init__.py
  model.py        # fake layers and minimal model
  schedule.py     # schedule data structures
  scheduler.py    # runtime scheduler
  training.py     # PyTorch-like training loop invoking scheduler hooks
  main.py         # small demo script
```

Code size is intentionally small; the goal is to surface the **ordering** and **trigger points** of scheduler activity.

---

## 1. Model and Layers (`model.py`)

### `FakeLayer`

A simple layer with:

* `name`
* optional simulated latency
* `forward(self, x)` that logs execution, optionally `sleep`s, and returns `x`

### `FakeModel`

Holds an ordered list of layers:

```python
def forward(self, x):
    for layer in self.layers:
        x = layer.forward(x)
    return x
```

This models a minimal `nn.Module` with submodules.

---

## 2. Training Loop (`training.py`)

A single training step mirrors the PyTorch control flow:

```python
def train_one_step(model, scheduler, batch):
    scheduler.on_step_start()

    x = batch
    # Forward
    for layer in model.layers:
        scheduler.on_pre_forward(layer)
        x = layer.forward(x)
        scheduler.on_post_forward(layer)

    # Backward (simulated)
    for layer in reversed(model.layers):
        scheduler.on_pre_backward(layer)
        scheduler.on_post_backward(layer)

    scheduler.on_step_end()
```

This layout aligns with:

```
for batch in data_loader:
    out = model(batch)
    loss.backward()
    optimizer.step()
```

with the scheduler interposed at each module boundary.

---

## 3. Schedule Format (`schedule.py`)

A schedule is a declarative list describing the sequence of logical operations to perform:

```python
from collections import namedtuple
Step = namedtuple("Step", ["kind", "layer_name"])

forward_schedule = [
    Step("fetch", "L1"),
    Step("compute", "L1"),
    Step("release", "L1"),
    Step("fetch", "L2"),
    Step("compute", "L2"),
    Step("release", "L2"),
]
```

No timestamps or real timings—just an ordered specification.

---

## 4. Runtime Scheduler (`scheduler.py`)

The scheduler consumes the schedule and executes operations when hooks fire:

```python
class Scheduler:
    def __init__(self, forward_schedule):
        self.forward_schedule = list(forward_schedule)
        self.idx = 0

    def on_step_start(self):
        self.idx = 0
        print("[scheduler] step start")

    def on_pre_forward(self, layer):
        # Execute operations until the layer's "compute" entry
        self._run_until("compute", layer)

    def on_post_forward(self, layer):
        # Handle post-compute tasks such as release/prefetch
        self._run_after_compute(layer)

    def on_pre_backward(self, layer): pass
    def on_post_backward(self, layer): pass

    def on_step_end(self):
        print("[scheduler] step end")
```

In `_run_until` and `_run_after_compute`, the scheduler:

* Inspects the next schedule entries
* Executes `fetch`, `compute`, or `release` when the entry matches the current layer
* Advances its internal index

This shows how hook invocation drives the schedule.

---

## 5. Optional: Prefetching

The schedule may include `Step("prefetch", layer_name)` entries.
These indicate that a `fetch(layer)` is permitted earlier than required. A simple implementation may:

* simulate I/O latency with `time.sleep`
* issue prefetch operations during earlier layers’ hooks
* optionally use threads or `asyncio` to overlap prefetch with compute
* log start/end times to show potential overlap

---

## 6. Execution Trace and Observations

Running the demo highlights:

* The exact positions in the training loop where scheduler callbacks occur
* How the schedule dictates which operations fire at each hook
* How this structure parallels systems like DeepSpeed that wrap module calls with `pre_sub_module_forward_function` / `post_sub_module_backward_function`
* How prefetch entries allow fetch operations to be moved earlier to hide latency

This compact setup makes the relation between a schedule, runtime hooks, and layer execution behavior explicit and easy to inspect.
