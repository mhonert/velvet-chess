
This releases comes with a new neural network, search improvements and some fixes


Estimated strength increase: ~ 40 Elo standard chess / 60 Elo FRC

## Changes
- New neural network trained from v5.1.0 and v5.2.0dev self-play games
- Search improvements
- Fixed fathom compilation issue for no-popcount Windows build
- Improved tablebase probing
- Switched transposition table replacement scheme
- Make tablebase probes resilient to incomplete TB sets

## Statistics

Overview of match statistics from two Gauntlet with the same set of opponent engines.

### 5.2.0

| Move range | Game % | Win % | Draw % | Loss % | Rel. Strength |
|------------|--------|-------|--------|--------|---------------|
| <= 40      | 14.0   | 39.2  | 59.6   | 1.2    | +88           |
| 41..60     | 43.4   | 51.8  | 38.3   | 9.9    | +110          |
| 61..80     | 24.9   | 38.9  | 33.6   | 27.5   | +35           |
| 81..100    | 9.7    | 15.2  | 49.0   | 35.8   | -37           |
| 101..120   | 4.3    | 10.1  | 59.7   | 30.2   | -34           |
| > 120      | 3.9    |  6.9  | 72.6   | 20.5   | -18           |

Note: the relative strength calculation does not take the absolute Elo of the opponents into account

### 5.1.0

| Move range | Game % | Win % | Draw % | Loss % | Rel. Strength |
|------------|--------|-------|--------|--------|---------------|
| <= 40      | 14.5   | 43.6  | 54.5   | 1.9    | +93           |
| 41..60     | 43.0   | 45.6  | 38.2   | 16.2   | +78           |
| 61..80     | 25.8   | 28.2  | 35.2   | 36.5   | -17           |
| 81..100    | 9.5    | 12.7  | 44.5   | 42.8   | -62           |
| 101..120   | 3.7    |  4.6  | 62.4   | 33.0   | -52           |
| > 120      | 3.5    |  3.1  | 74.8   | 22.1   | -23           |


### Comparison

The main strength increase is in games with 61 to 80 moves, where Velvet v5.1.0 previously was -17 Elo below the 
average opponent in the set and v5.2.0 is now at +37 Elo.

Also the range of 41..60 moves has clearly improved: v5.1.0 with +78 vs. v5.20 with +110 Elo.
In games with increasing number of move numbers (81 moves and more), Velvet is still weaker than the average opponent in the Gauntlet,
but the situation improved.

The strength in games with 40 moves and less is still very high, but did not improve with the new version.
There is even a slight reduction, which is within the error bounds, but could also indicate a minor regression in that area.

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
