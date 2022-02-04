
This release improves the neural network based evaluation

## Changes
- Neural network trainer
  - fixed a bug which prevented successful training of networks with bigger hidden layers
  - added data compression support for trained network files
  
- Neural network evaluation
  - increased hidden layer size from 64 to 256
  - implemented some optimizations:
      - defer updates and calculations until *eval* is called
      - remove additional hidden layers

## Installation
The chess engine is available for Windows and Linux and requires a 64 Bit CPU.
There are optimized executables available for different CPU micro-architecture generations.

If you have a relatively modern CPU (2013+) with AVX2 support, then the *...-x86_64-avx2* executable is highly recommended for best performance.

| Executable          | Description                                                       | Min. CPU Generation           | Required Instruction Sets |
| ------------------- | ----------------------------------------------------------------- | ----------------------------- | ------------------------- |
| x86_64-avx2         | Recommended for best performance on a modern CPU                  | Intel Haswell / Zen1          | AVX2, BMI1                |
| x86_64-sse4-popcnt  | Lower performance, recommended for CPUs without AVX2 support      | Intel Nehalem / AMD Bulldozer | SSE4.2, SSE3, POPCNT      |
| x86_64-nopopcnt     | Lowest performance, but compatible with most x86_64 CPUs          | ---                           | SSE2, CMOV                |
