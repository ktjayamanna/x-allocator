"""
Profile the full training loop to detect:
1. Non-contiguous tensors in model forward pass
2. GPU idle time during data transfer

This demonstrates the unified profiler that tracks both contiguity events
and GPU idle periods.
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config
from utils import load_and_prepare_data, get_model
from profiler import ContiguityProfiler


def main():
    print("="*80)
    print("Training Loop Profiler - Unified Contiguity + GPU Idle Time Profiling")
    print("="*80)
    print()
    
    # Load data and model
    train_dataset, _ = load_and_prepare_data()
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )
    
    model = get_model(vocab_size=train_dataset.vocab_size)
    model.train()
    
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # Define training step function
    def train_step(x, y):
        """Single training step: forward, loss, backward, optimizer step."""
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Create profiler
    profiler = ContiguityProfiler(
        model, 
        device=config.DEVICE,
        measure_conversion_cost=True,
        sample_conversion_for_all_shapes=True
    )
    
    print("Profiling training loop...")
    print(f"  Device: {config.DEVICE}")
    print(f"  Warmup iterations: 2")
    print(f"  Profile iterations: 5")
    print()

    # Profile the full training loop
    profiler.profile(
        dataloader=train_loader,
        train_step_fn=train_step,
        warmup=2,
        iters=5
    )
    
    print("Profiling complete!")
    print()
    
    # Display summary
    profiler.summarize(top_k=10)
    
    # Export to JSON
    output_path = os.path.join(config.TMP_DIR, "training_loop_profile.json")
    os.makedirs(config.TMP_DIR, exist_ok=True)
    profiler.export_json(output_path)
    
    print()
    print("="*80)
    print("Key Insights:")
    print("="*80)
    
    # Analyze results
    if profiler.idle_events:
        avg_idle = sum(e.duration_ms for e in profiler.idle_events) / len(profiler.idle_events)
        print(f"✓ GPU Idle Time (data transfer): {avg_idle:.2f} ms per iteration")
        print(f"  This is the window available for scheduling .contiguous() conversions")
    
    if profiler.records:
        total_forward = sum(r.forward_time_ms for r in profiler.records)
        print(f"✓ Total Forward Pass Time: {total_forward:.2f} ms")
        
        noncontig_records = [r for r in profiler.records if r.has_noncontig_input or r.has_noncontig_output]
        if noncontig_records:
            total_conv_cost = sum(
                r.estimated_conversion_cost_ms 
                for r in noncontig_records 
                if r.estimated_conversion_cost_ms is not None
            )
            print(f"✓ Total Conversion Cost: {total_conv_cost:.2f} ms")
            print(f"  Number of modules with non-contiguous tensors: {len(noncontig_records)}")
    
    print()
    print(f"Full profile saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()

