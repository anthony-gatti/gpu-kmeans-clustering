# GPU-Accelerated K-Means Clustering

This project explores multiple implementations of the K-Means clustering algorithm, with a focus on GPU acceleration. It benchmarks performance across datasets of varying size and dimensionality, highlighting the tradeoffs between convenience, flexibility, and raw performance.

## Overview

K-Means clustering is a widely-used unsupervised machine learning algorithm that partitions data into *K* clusters by minimizing intra-cluster variance. This project implements and compares:

- A **sequential (CPU)** baseline
- A parallel version using **OpenACC**
- A highly tuned **KM-CUDA** GPU library
- A custom **CUDA kernel** implementation

These implementations are evaluated on their ability to scale with increasing dataset size and cluster count (*K*), with special attention to performance bottlenecks in memory management and thread synchronization.

## Implementations

- `serial_kmeans.cpp`: Sequential CPU implementation.
- `openacc_kmeans.cpp`: OpenACC-accelerated implementation with automatic GPU parallelization.
- `kmcuda_interface.cpp`: Wrapper to interface with the KM-CUDA library.
- `cuda_kmeans.cu`: Handwritten CUDA kernel implementation for full control over memory layout and execution.

Each version is built using `make`, with options to toggle between implementations via preprocessor flags.

## Usage

### Compilation

```bash
make
```

### Benchmarking
```bash
python3 benchmark.py
```
This script runs all versions of the K-Means algorithm across various configurations and records the execution time for comparison.

### Plotting Results
```bash
python3 plot.py
```
This generates performance graphs that compare the different implementations over:
- Varying K values
- Dataset size (rows × features)
- Dimensionality

## Benchmark Highlights
- **KM-CUDA** is the fastest for most large datasets and higher K values, benefiting from optimized production-level code.
- **Handwritten CUDA kernels** are competitive and sometimes outperform KM-CUDA on high-dimensional datasets, thanks to careful shared memory use and loop unrolling.
- **OpenACC** is effective on small datasets or low K values but scales poorly due to limited control over memory and thread synchronization.
- **Serial implementation** serves as a performance baseline and predictably lags behind for larger workloads.

## Notes
- Performance graphs are available via the plot.py script, though raw benchmark data is not included in this repository.
- Profiling was conducted using NVIDIA Nsight to optimize kernel execution and memory access patterns.