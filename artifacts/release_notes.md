
The major feature of this new release is the addition of a neural network for position evaluation :robot:

The neural network has been trained from a set of 800 million chess positions from Velvet self-play games.
For this I created two new tools:
- **gensets**: Extracts and labels chess positions from games that Velvet plays against itself
- **trainer**: Trains a neural network using the labeled chess position sets as training data 

## Changes
- Added neural network based evaluation
- Optimized slider attack generation using magic bitboards
- Refactored and simplified time management
- Additional optimizations and minor bug-fixes

## Installation
The chess engine is available for Windows and Linux and requires a 64 Bit CPU.
There are optimized executables available for different CPU micro-architecture generations.

If you have a relatively modern CPU (2013+) with AVX2 support, then the *...-x86_64-avx2* executable is highly recommended for best performance.

| Executable          | Description                                                       | Min. CPU Generation           | Required Instruction Sets |
| ------------------- | ----------------------------------------------------------------- | ----------------------------- | ------------------------- |
| x86_64-avx2         | Recommended for best performance on a modern CPU                  | Intel Haswell / Zen1          | AVX2, BMI1                |
| x86_64-sse4-popcnt  | Lower performance, recommended for CPUs without AVX2 support      | Intel Nehalem / AMD Bulldozer | SSE4.2, SSE3, POPCNT      |
| x86_64-nopopcnt     | Lowest performance, but compatible with most x86_64 CPUs          | ---                           | SSE2, CMOV                |
