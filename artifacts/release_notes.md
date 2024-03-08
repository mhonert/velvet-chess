
This is just a minor release with some bug-fixes, performance improvements and a small strength increase

Estimated strength increase: ~ 20 Elo

## Changes
- Search improvements
- Performance optimizations

## Fixed bugs
- UCI option handling: support spaces inside the 'value' part
- 50-move draw rule logic: checkmate at 100th half-move evaluated as draw

## Statistics
- Elo change: v7.1.0 compared to v7.0.0 against the same set of opponents
- Move range: grouped by games won in less than x moves (each game only belongs to one group, so a game that ended in 57 moves would belong to the group "60", but not "80", "100", etc.)

| Move range | Elo change |
|------------|------------|
| 40         | +42        |
| 60         | +9         |
| 80         | +28        |
| 100        | +31        |
| 120        | +26        |
| \>= 120    | +2         |

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
