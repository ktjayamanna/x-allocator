"""
Schedule Generation - Step 6

Greedy strategy: Start each operation as early as possible

Vocabulary: {prefetch, wait, compute, release}
Constraints:
  - Hard: prefetch(N) before wait(N)
  - Hard: wait(N) before compute(N)
  - Soft: Minimize total time
"""

from collections import namedtuple

Step = namedtuple("Step", ["kind", "batch_id"])

def generate_schedule_greedy(num_batches):
    """
    Generate schedule using greedy approach: start each operation ASAP

    Strategy:
    1. First batch: prefetch → wait → compute (blocking, can't overlap)
    2. Remaining batches: compute(N) + prefetch(N+1) → release(N) → wait(N+1)
       (overlap prefetch with compute to hide I/O latency)

    Args:
        num_batches: Number of batches to process

    Returns:
        List of Step(kind, batch_id)
    """
    schedule = []

    # First batch: Must prefetch and wait before we can compute
    schedule.append(Step("prefetch", 0))
    schedule.append(Step("wait", 0))

    for batch_id in range(num_batches):
        # Compute current batch
        schedule.append(Step("compute", batch_id))

        # Greedy: Start next prefetch ASAP (right after compute starts)
        # This overlaps prefetch(N+1) with compute(N)
        if batch_id < num_batches - 1:
            schedule.append(Step("prefetch", batch_id + 1))

        # Release current batch (free GPU memory)
        schedule.append(Step("release", batch_id))

        # Wait for next batch to be ready (should already be done if prefetch < compute)
        if batch_id < num_batches - 1:
            schedule.append(Step("wait", batch_id + 1))

    return schedule


def print_schedule(schedule, max_steps=20):
    """Pretty print the schedule"""
    print(f"Schedule ({len(schedule)} steps):")

    for i, step in enumerate(schedule[:max_steps]):
        print(f"  {i:3d}: {step.kind:8s} batch_{step.batch_id}")

    if len(schedule) > max_steps:
        print(f"  ... ({len(schedule) - max_steps} more steps)")


def analyze_schedule(schedule, costs):
    """
    Analyze schedule performance given operation costs

    Args:
        schedule: List of Step(kind, batch_id)
        costs: Dict of {operation_name: time_in_ms}

    Returns:
        Dict with analysis results
    """
    # Count operations
    op_counts = {}
    for step in schedule:
        op_counts[step.kind] = op_counts.get(step.kind, 0) + 1

    # Calculate time with overlap (prefetch happens during compute)
    # First batch: prefetch + wait + compute + release
    # Remaining batches: compute + release (prefetch overlapped)
    num_batches = max(step.batch_id for step in schedule) + 1

    first_batch_time = costs["prefetch"] + costs["wait"] + costs["compute"] + costs["release"]
    remaining_batch_time = costs["compute"] + costs["release"] + costs["wait"]
    total_time_optimized = first_batch_time + (num_batches - 1) * remaining_batch_time

    # Baseline (no overlap)
    baseline_time = num_batches * (costs["prefetch"] + costs["wait"] + costs["compute"] + costs["release"])

    speedup = (baseline_time - total_time_optimized) / baseline_time * 100

    return {
        "num_batches": num_batches,
        "num_steps": len(schedule),
        "op_counts": op_counts,
        "baseline_time_ms": baseline_time,
        "optimized_time_ms": total_time_optimized,
        "speedup_percent": speedup,
        "time_per_batch_ms": total_time_optimized / num_batches,
    }


if __name__ == "__main__":
    # Example: Generate schedule for 5 batches
    schedule = generate_schedule_greedy(num_batches=5)
    print_schedule(schedule)

    # Example costs (from cost profiling)
    costs = {
        "prefetch": 0.1,    # ms
        "wait": 0.008,      # ms
        "compute": 211.6,   # ms
        "release": 0.030,   # ms
    }

    print("\n" + "=" * 60)
    analysis = analyze_schedule(schedule, costs)

    print(f"\nSchedule Analysis:")
    print(f"  Batches:           {analysis['num_batches']}")
    print(f"  Total steps:       {analysis['num_steps']}")
    print(f"  Operations:        {analysis['op_counts']}")
    print(f"\nPerformance:")
    print(f"  Baseline (no overlap):  {analysis['baseline_time_ms']:.2f}ms")
    print(f"  Optimized (overlap):    {analysis['optimized_time_ms']:.2f}ms")
    print(f"  Speedup:                {analysis['speedup_percent']:.2f}%")
    print(f"  Time per batch:         {analysis['time_per_batch_ms']:.2f}ms")
