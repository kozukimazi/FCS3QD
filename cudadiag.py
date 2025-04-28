import numpy as np
import cupy as cp
import time

# Matrix size (adjust to test performance)
n = 16600

# Generate a random symmetric matrix (for real eigenvalues)
A_cpu = np.random.rand(n, n)
A_cpu = A_cpu + A_cpu.T  # Make symmetric
A_gpu = cp.array(A_cpu)  # Copy to GPU

# NumPy (CPU) diagonalization
start = time.time()
eigvals_cpu, eigvecs_cpu = np.linalg.eigh(A_cpu)
cpu_time = time.time() - start
print(f"NumPy (CPU) time: {cpu_time:.4f} seconds")

# CuPy (GPU) diagonalization
start = time.time()
eigvals_gpu, eigvecs_gpu = cp.linalg.eigh(A_gpu)
cp.cuda.Stream.null.synchronize()  # Wait for GPU to finish
gpu_time = time.time() - start
print(f"CuPy (GPU) time: {gpu_time:.4f} seconds")

# Speedup ratio
print(f"Speedup: {cpu_time / gpu_time:.2f}x")