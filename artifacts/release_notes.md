
This is just a preview release not intended for rating lists

## Changes

- Switch from 16- to 8-bit quantized weights in the input layer
- Search improvements
- Support for AVX-512
- New SPRT testing tool
- New tournament (gauntlet) testing tool
- New tuning tool
- Changed default value for UCI option `RatingAdvAdaptiveStyle` from `true` to `false`

## Notes

Due to the lack of an ARM-based (Apple Silicon) computer, the "apple-silicon" builds are untested.

## Installation
The chess engine is available for Windows and Linux and requires a 64 Bit CPU.
There are optimized executables available for different CPU micro-architecture generations.

Starting with Velvet v4.1.0 there are also builds for macOS provided.
Currently there are no specific optimizations for the ARM-based/Apple Silicon builds implemented, so
the macOS builds for x86_64 might be faster.

If you have a relatively modern CPU (2013+) with AVX2 support, then the *...-x86_64-avx2* executable is highly recommended for best performance.

| Executable         | Description                                                                  | Min. CPU Generation           | Required Instruction Sets |
|--------------------|------------------------------------------------------------------------------|-------------------------------|---------------------------|
| x86_64-avx512      | Higher performance for CPUs supporting AVX-512                               | x86-64-v4                     | AVX-512                   |
| x86_64-avx2        | Recommended for best performance on most modern CPUs without AVX-512 support | Intel Haswell / Zen1          | AVX2, BMI1                |
| x86_64-sse4-popcnt | Lower performance, recommended for CPUs without AVX2 support                 | Intel Nehalem / AMD Bulldozer | SSE4.2, SSE3, POPCNT      |
| x86_64-nopopcnt    | Lowest performance, but compatible with most x86_64 CPUs                     | ---                           | SSE2, CMOV                |
| apple-silicon      | Native builds for Apple Silicon processors (ARM aarch64)                     | Apple M1                      |                           |
