
The main focus of this release was to experiment with a new neural network architecture

In order to allow the training of bigger neural networks, I implemented a new trainer which uses multiple CPU cores
instead of a GPU. There is still room for optimizations, but the iteration time was sufficiently low to experiment
with different neural network architectures and sizes.

Self-play strength increase: ~ 40 Elo / FRC: 100 Elo
(Note: strength increase is probably lower against a range of different opponents)

## Changes
- New trainer to train bigger networks in a reasonable amount of time
  
- New neural network architecture
  - Input nodes: 768x24 instead of 768x5
  - Hidden nodes: 2x288 instead of 512
  
- Some time management adjustments

## Installation
The chess engine is available for Windows and Linux and requires a 64 Bit CPU.
There are optimized executables available for different CPU micro-architecture generations.

If you have a relatively modern CPU (2013+) with AVX2 support, then the *...-x86_64-avx2* executable is highly recommended for best performance.

| Executable          | Description                                                       | Min. CPU Generation           | Required Instruction Sets |
| ------------------- | ----------------------------------------------------------------- | ----------------------------- | ------------------------- |
| x86_64-avx2         | Recommended for best performance on a modern CPU                  | Intel Haswell / Zen1          | AVX2, BMI1                |
| x86_64-sse4-popcnt  | Lower performance, recommended for CPUs without AVX2 support      | Intel Nehalem / AMD Bulldozer | SSE4.2, SSE3, POPCNT      |
| x86_64-nopopcnt     | Lowest performance, but compatible with most x86_64 CPUs          | ---                           | SSE2, CMOV                |
