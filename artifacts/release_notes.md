
This minor release provides support for pondering and an estimated strength increase of about 50 Elo.

## Changes
- New engine feature: Pondering support
- Search improvements:
  - Singular extensions
  - Disable null move pruning in pawn endgames
  - SEE and history based pruning of quiet moves
  - Update counter/killer moves for TT cut-offs
  - Some small bug fixes and improvements
  
- Technical changes:
  - New tooling for automated patch verification
  - Removed dependency on packed_simd library, which required Rust Nightly builds
  - Switched back to Rust Stable
  - Use PGO (profile-guided optimization) builds in pipeline
  - Increased search depth limit
  - Some minor refactorings

## Installation
The chess engine is available for Windows and Linux and requires a 64 Bit CPU.
There are optimized executables available for different CPU micro-architecture generations.

If you have a relatively modern CPU (2013+) with AVX2 support, then the *...-x86_64-avx2* executable is highly recommended for best performance.

| Executable          | Description                                                       | Min. CPU Generation           | Required Instruction Sets |
| ------------------- | ----------------------------------------------------------------- | ----------------------------- | ------------------------- |
| x86_64-avx2         | Recommended for best performance on a modern CPU                  | Intel Haswell / Zen1          | AVX2, BMI1                |
| x86_64-sse4-popcnt  | Lower performance, recommended for CPUs without AVX2 support      | Intel Nehalem / AMD Bulldozer | SSE4.2, SSE3, POPCNT      |
| x86_64-nopopcnt     | Lowest performance, but compatible with most x86_64 CPUs          | ---                           | SSE2, CMOV                |
