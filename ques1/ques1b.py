import math
from tabulate import tabulate

# ======================= Part (a): Single GPU =======================

# Model Specification
NUM_LAYERS       = 32
NUM_QUERY_HEADS  = 32
NUM_KV_HEADS     = 8
EMBED_DIM        = 4096
INNER_DIM        = 14336
VOCAB_SIZE       = 128256
GPU_MEMORY_GB    = 80.0  # 80GB 

# Precision & KV cache
WEIGHTS_BYTES    = 2  # 16-bit precision
KV_CACHE_BYTES   = 2  # 16-bit precision
CONTEXT_LENGTH   = 1024  

# Computing total model parameters
embedding_params = VOCAB_SIZE * EMBED_DIM
ffn_params_per_layer = 2 * INNER_DIM * EMBED_DIM + INNER_DIM + EMBED_DIM
total_ffn_params = NUM_LAYERS * ffn_params_per_layer

head_dim = EMBED_DIM // NUM_QUERY_HEADS  # Per-head dimension
kv_dim_total = NUM_KV_HEADS * head_dim   # Total KV dimension

attention_params_per_layer = EMBED_DIM * (EMBED_DIM + 2 * kv_dim_total)
total_attention_params = NUM_LAYERS * attention_params_per_layer

output_layer_params = VOCAB_SIZE * EMBED_DIM

# Summing everything
total_params = embedding_params + total_ffn_params + total_attention_params + output_layer_params

# Computing total parameter memory (GB)
total_param_mem_bytes = total_params * WEIGHTS_BYTES
total_param_mem_gb = total_param_mem_bytes / (1024 ** 3)

# KV Cache memory per request (batch size = 1)
kv_dim_per_token = NUM_KV_HEADS * head_dim * 2  # For K & V
total_kv_bytes = kv_dim_per_token * CONTEXT_LENGTH * NUM_LAYERS * KV_CACHE_BYTES
kv_cache_gb = total_kv_bytes / (1024 ** 3)

# Computing max batch size for single GPU
leftover_mem_gb = GPU_MEMORY_GB - total_param_mem_gb
max_batch_size = math.floor(leftover_mem_gb / kv_cache_gb) if kv_cache_gb > 0 else 0


# ======================= Part (b): Tensor Parallelism =======================

# Defining the Tensor Parallelism (TP) dimension
TP_DIM = 8  

# Computing parameter memory per GPU
param_mem_per_gpu = total_param_mem_gb / TP_DIM

# Computing KV cache memory per GPU (for batch size = 1)
kv_cache_per_gpu = kv_cache_gb / TP_DIM

# Computing max batch size per GPU
leftover_mem_gpu = GPU_MEMORY_GB - param_mem_per_gpu
max_batch_size_tp = math.floor(leftover_mem_gpu / kv_cache_per_gpu) if kv_cache_per_gpu > 0 else 0

# Printing results for Part (b)
print(f"\n===== Part (b): Tensor Parallelism (TP={TP_DIM}) - Llama3–8B =====")
print(tabulate([
    ["Total parameter memory per GPU (GB)", f"{param_mem_per_gpu:.3f}"],
    ["KV cache per GPU (GB)", f"{kv_cache_per_gpu:.3f}"],
    ["Max batch size per GPU", str(max_batch_size_tp)]
], headers=["Item", "Value"], tablefmt="grid"))
