## Producer-Consumer Relationship Algorithm (TensorFingerprint-based)

The algorithm forms producer-consumer relationships using a **three-field fingerprint** (anchor_name, object_id, data_ptr) to robustly identify tensors as they flow through the computation graph:

### During Forward Hook Execution:

- **For each output tensor** of an operation:
  1. Create a `TensorFingerprint` with anchor_name derived from explicit Mark name or module registry name
  2. Generate a unique fingerprint key: `"{anchor_name}|{object_id}|{data_ptr}"`
  3. Register the tensor as produced by current operation: `fingerprint_producers[fp_key] = (fingerprint, op_id)`
  4. Store a cross-reference: `_tensor_to_fingerprint[(object_id, data_ptr)] = fp_key`

- **For each input tensor** of an operation:
  1. Look up `(object_id, data_ptr)` in the cross-reference map
  2. If found, retrieve the fingerprint key and original producer
  3. Register current operation as a consumer: `fingerprint_consumers[fp_key].append((fingerprint, current_op_id))`

### Building the Flow Graph:

- **Edges are formed** when: an input tensor's `(object_id, data_ptr)` matches a previously seen output tensor
- Each edge records: `(producer_op_id, consumer_op_id, fingerprint_key, anchor_name)`
