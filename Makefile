# Compiler settings
CXX = g++
NVCC = nvcc  # NVIDIA CUDA Compiler

CXXFLAGS = -O3 -std=c++17 -fopenacc -fopenmp -I../kmcuda/src
NVCCFLAGS = -arch=sm_60 -O3 -std=c++17 -I../kmcuda/src

LDFLAGS = -L../kmcuda/build -lKMCUDA -Xlinker -rpath=../kmcuda/build

# List of executables
TARGETS = kmeans-serial kmeans-gpu-v1 kmeans-gpu-v2 kmeans-gpu-v3

# Default target: build all executables.
all: $(TARGETS)

# Serial version: compiled with g++
kmeans-serial: src/kmeans-serial.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# GPU version: compiled with nvcc
kmeans-gpu-v1: src/kmeans-gpu-v1.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# GPU version v2: compiled with nvcc
kmeans-gpu-v2: src/kmeans-gpu-v2.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# GPU version v3: compiled with nvcc
kmeans-gpu-v3: src/kmeans-gpu-v3.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

# Run target: run all executables with dataset3.txt.
run: $(TARGETS)
	@echo "Running all executables with dataset3.txt..."
	@for exe in $(TARGETS); do \
		echo "Running $$exe:"; \
		./$$exe < datasets/dataset3.txt; \
		echo ""; \
	done

# Clean target: remove all executables
clean:
	rm -f $(TARGETS)

.PHONY: all run clean