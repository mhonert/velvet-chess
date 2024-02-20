
This releases comes with a new neural network architecture and some small search improvements :christmas_tree:

Estimated strength increase: ~ 65 Elo

## New Features
- Support for UCI "Move Overhead" option

## Changes
- Increase default move overhead from 16 to 20 milliseconds
- Performance optimizations
- Cache evaluation scores in transposition table / Up to 50% higher performance in positions with fewer pieces
- Rescaled evaluation scores in training data
- Store scores internally with additional bit: eval range now from -9999 to 9999 (was -7650 to 7650 before)
- Perform check evasions in quiescence search
- Collect PV (for UCI output) also during quiescence search
- Several search improvements

## Fixed bugs
- Eval scores reaching into mate score range
- Time management issue causing search to be stopped too early

## Statistics

- Elo change: v7.0.0 compared to v6.0.0 against the same set of opponents
- Move range: grouped by games won in less than x moves (each game only belongs to one group, so a game that ended in 57 moves would belong to the group "60", but not "80", "100", etc.)

| Move range | Elo change |
|------------|------------|
| 40         | +4         |
| 60         | +51        |
| 80         | +88        |
| 100        | +100       |
| 120        | +88        |
| \>= 120    | +33        |

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
