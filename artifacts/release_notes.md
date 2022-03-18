
This release provides support for Fischer Random Chess (Chess960) and a small strength increase

Estimated strength increase: ~ 30 Elo

## Changes
- Support Fischer random chess (Chess960)
  
- Neural network evaluation
  - increased hidden layer size from 256 to 512
  - trained new network (including Chess960 positions)
  
- Some refactorings and minor search improvements

## Installation
The chess engine is available for Windows and Linux and requires a 64 Bit CPU.
There are optimized executables available for different CPU micro-architecture generations.

If you have a relatively modern CPU (2013+) with AVX2 support, then the *...-x86_64-avx2* executable is highly recommended for best performance.

| Executable          | Description                                                       | Min. CPU Generation           | Required Instruction Sets |
| ------------------- | ----------------------------------------------------------------- | ----------------------------- | ------------------------- |
| x86_64-avx2         | Recommended for best performance on a modern CPU                  | Intel Haswell / Zen1          | AVX2, BMI1                |
| x86_64-sse4-popcnt  | Lower performance, recommended for CPUs without AVX2 support      | Intel Nehalem / AMD Bulldozer | SSE4.2, SSE3, POPCNT      |
| x86_64-nopopcnt     | Lowest performance, but compatible with most x86_64 CPUs          | ---                           | SSE2, CMOV                |
