
This releases comes with a new neural network architecture and some small search improvements :christmas_tree:

Estimated strength increase: ~ 20 Elo

## Changes
- Change neural network architecture from 3x5x768->2x512->1 to 32x768->2x512->1
- New neural network trained from v5.3.0 (and v6.0.0 dev) self-play games
- Updated training tools
- Some refactorings and optimizations

## Statistics

- Elo change: v6.0.0 compared to v5.3.0 against the same set of opponents
- Move range: grouped by games won in less than x moves (each game only belongs to one group, so a game that ended in 57 moves would belong to the group "60", but not "80", "100", etc.)

| Move range | Elo change |
|------------|------------|
| 40         | +26        |
| 60         | +7         |
| 80         | +21        |
| 100        | +52        |
| 120        | +46        |
| \>= 120    | +40        |

## Notes

Due to the lack of an ARM-based (Apple Silicon) computer, the new "apple-silicon" builds are untested.

## Installation
The chess engine is available for Windows and Linux and requires a 64 Bit CPU.
There are optimized executables available for different CPU micro-architecture generations.

Starting with Velvet v4.1.0 there are also builds for macOS provided.
Currently there are no specific optimizations for the ARM-based/Apple Silicon builds implemented, so
the macOS builds for x86_64 might be faster.

If you have a relatively modern CPU (2013+) with AVX2 support, then the *...-x86_64-avx2* executable is highly recommended for best performance.

| Executable          | Description                                                       | Min. CPU Generation           | Required Instruction Sets |
| ------------------- | ----------------------------------------------------------------- | ----------------------------- | ------------------------- |
| x86_64-avx2         | Recommended for best performance on a modern CPU                  | Intel Haswell / Zen1          | AVX2, BMI1                |
| x86_64-sse4-popcnt  | Lower performance, recommended for CPUs without AVX2 support      | Intel Nehalem / AMD Bulldozer | SSE4.2, SSE3, POPCNT      |
| x86_64-nopopcnt     | Lowest performance, but compatible with most x86_64 CPUs          | ---                           | SSE2, CMOV                |
| apple-silicon       | Native builds for Apple Silicon processors (ARM aarch64)          | Apple M1                      |                           |
