import matplotlib.pyplot as plt

# Computing Arithmetic Intensity for prefill attention (batch=1).
def compute_ai_prefill(seq_length, model_dim, dtype_size=2):
    operations = 4 * (seq_length**2) * model_dim + 5 * (seq_length**2)
    memory_accesses = 4 * seq_length * model_dim + 3 * (seq_length**2) + seq_length * model_dim
    memory_bytes = memory_accesses * dtype_size
    return operations / memory_bytes if memory_bytes > 0 else 0

# Computing Arithmetic Intensity for single-step decode attention.
def compute_ai_decode(seq_length, model_dim, dtype_size=2):
    operations = 4 * seq_length * model_dim + 5 * seq_length
    memory_accesses = (2 * seq_length + 1) * model_dim + 3 * seq_length
    memory_bytes = memory_accesses * dtype_size
    return operations / memory_bytes if memory_bytes > 0 else 0

# Computing Arithmetic Intensity for batched decode attention
def compute_ai_decode_batch(seq_length, model_dim, batch_size, dtype_size=2):
    operations = batch_size * (4 * seq_length * model_dim + 5 * seq_length)  
    memory_accesses = (2 * seq_length + 1) * model_dim + 3 * seq_length * batch_size  
    memory_bytes = memory_accesses * dtype_size
    return operations / memory_bytes if memory_bytes > 0 else 0

# main function to calculate AI and plot graphs 
def analyze_arithmetic_intensity():
    print("========== Arithmetic Intensity Analysis ==========")
    
    seq_fixed = 1024
    dim_fixed = 128
    data_precision = 2  # 2 bytes per element (FP16)

    #Computing AI for a specific scenario
    prefill_ai_fixed = compute_ai_prefill(seq_fixed, dim_fixed, data_precision)
    decode_ai_fixed  = compute_ai_decode(seq_fixed, dim_fixed, data_precision)

    print(f"For seq_length={seq_fixed}, model_dim={dim_fixed}, FP16 (2 bytes/element):")
    print(f"  Prefill AI = {prefill_ai_fixed:.3f} FLOPs/byte")
    print(f"  Decode  AI = {decode_ai_fixed:.3f} FLOPs/byte")

    #AI vs. Sequence Length for Prefill
    seq_values = [128, 256, 512, 1024, 2048, 4096]
    prefill_ai_values = [compute_ai_prefill(n, dim_fixed, data_precision) for n in seq_values]

    #AI vs. Batch Size for Decode 
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    decode_ai_batch = [compute_ai_decode_batch(seq_fixed, dim_fixed, B, data_precision) for B in batch_sizes]

    # Plot Prefill AI vs. Sequence Length
    plt.figure(figsize=(8,5))
    plt.plot(seq_values, prefill_ai_values, marker='o', linestyle='-', color='blue', label="Prefill AI")
    plt.title("Prefill AI vs. Sequence Length (batch=1)")
    plt.xlabel("Sequence Length (N)")
    plt.ylabel("Arithmetic Intensity (FLOPs/Byte)")
    plt.grid(True)
    plt.legend()
    plt.savefig("prefill_ai_vs_seq_length.png")
    plt.show()

    # Plot Decode AI vs. Batch Size
    plt.figure(figsize=(8,5))
    plt.plot(batch_sizes, decode_ai_batch, marker='s', linestyle='-', color='red', label="Decode AI")
    plt.title("Decode AI vs. Batch Size")
    plt.xlabel("Batch Size (B)")
    plt.ylabel("Arithmetic Intensity (FLOPs/Byte)")
    plt.yscale("log") 
    plt.grid(True)
    plt.legend()
    plt.savefig("decode_ai_vs_batch_size.png")
    plt.show()

# Execute the analysis
analyze_arithmetic_intensity()
