## Tensor Lifetime Classification System

### Completed Implementation

#### 1. Updated `Mark` Module
The `Mark` module now enforces a mandatory name argument:

```python
class Mark(nn.Module):
    """
    Pass-through module that tags a tensor with a user-defined name.
    Implemented as part of the Explicit Naming Contract migration.
    """
    def forward(self, x, name: str):
        # The hook captures the 'name' string from inputs
        return x
```

#### 2. Hybrid Profiling Strategy (Active)
The system captures a comprehensive list of all non contiguous tensors using our hybrid approach:
- **is_contiguous**: If the tensor is contiguous, all the following steps are skipped.
- **Explicit Names (Priority)**: When hooks detect a string argument from `Mark` modules, they use it as the **Anchor Key**
- **Implicit Names (Fallback)**: For standard layers like `nn.Conv2d`, the system falls back to the module's unique registry name (e.g., `layers.0.conv1`) as the **Anchor Key**

#### 3. Three-Field Persistence Check (Operational)
Every non contiguous tensor is tracked across two distinct iterations (batches). For each Anchor Key, we maintain a fingerprint with three fields:

1. **Anchor Name** (String ID)
2. **Python Object ID** (`id(tensor)`)
3. **Data Pointer** (`tensor.data_ptr()`)

### Current Decision Logic
The system compares fingerprints between Batch N and Batch N+1:

**CASE A: Persistent Tensors (Optimization Targets)**
- Name matches AND
- Object ID matches AND
- Data pointer matches
- **Action**: Tensor is marked as persistent (stable).

**CASE B: Transient Tensors (Ignored)**
- Name matches BUT
- Object ID changed OR Data pointer changed
- **Action**: Tensor is identified as transientephemeral.
