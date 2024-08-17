
This new major release introduces configurable playing strength, adaptive styles, and expanded platform support

Estimated strength increase (using the default network): ~ 35 Elo

## New features

### Configurable strength

In order to make Velvet a viable sparring partner for human players of different skill levels, its strength can now be limited.
To enable this feature, set `UCI_LimitStrength` to true and adjust `UCI_Elo` to your desired Elo rating (between 1225 and 3000). 

Please note that these Elo ratings might not yet correspond well to Elo ratings of human players.
A better calibration would require a lot of games against human players of different skill levels.
I recommend to experiment with different settings to find the optimal match for your current skill.

In addition, the new `SimulateThinkingTime` option allows Velvet to mimic human-like thinking times by utilizing a portion of its remaining time budget. 
You can disable this feature by setting `SimulateThinkingTime` to false.

These options can be combined with the new "style" settings.

### Support for different playing styles

Velvet now offers two distinct embedded neural networks, each designed to reflect a different playing style. 
You can easily toggle between these styles using the `Style` UCI option to match your strategic preferences.

* **Normal Style**: The default setting, offering a balanced approach to gameplay. While Velvet is still capable of sacrifices and aggressive attacks, it places slightly more emphasis on avoiding unfavorable positions if an attack doesn’t succeed, compared to the Risky style.
* **Risky Style**: This setting pushes Velvet to adopt a bolder, more aggressive approach, taking greater risks to create dynamic and challenging positions on the board.

Note: the riskier playing style comes with the downside, that the strength is reduced by around 25 Elo depending upon the opponent.

### New `bench` Command for Performance Benchmarking

Velvet now includes support for the `bench` command, allowing users to perform a quick benchmark test to evaluate the engine's performance. 
This command runs searches from several predefined positions and reports key metrics, 
including the total number of nodes processed, the duration of the test, and the nodes per second (NPS) achieved.

Example output:

```
bench
[...]

info string bench total time    : 2005ms
info string bench nodes         : 5199862
info string bench NPS           : 2593447
```

### Support for UCI_RatingAdv and UCI_Opponent options

Velvet now supports the `UCI_RatingAdv` and `UCI_Opponent` options, allowing the engine to dynamically adjust its playing style based on the set rating advantage. 
When Velvet detects a significant rating advantage, it will adopt a more aggressive, risk-taking approach.

This is mainly relevant for the TCEC event, where these two options are provided to the engines.

If desired, this adaptive behavior can be disabled by setting the `RatingAdvAdaptiveStyle` option to `false`.

### New 64-bit Executable for Raspberry Pi Support

Velvet now includes a 64-bit executable for the Linux aarch64 architecture 
to support for devices like the Raspberry Pi with 64-bit capabilities (e.g., Raspberry Pi 3, Raspberry Pi Zero 2, and similar models).

## Additional Changes

In addition to the new features, Velvet has undergone several internal improvements:

- **Search Enhancements:** Various optimizations to improve search efficiency and accuracy.
- **Updated Rust Version:** Velvet has been updated to Rust 1.80.0
- **Neural Network Upgrade:** The size of the hidden layer has been increased from 2x768 to 2x1024
- **Improved Training Data:** Adjustments have been made to training data, including reduced scores for positions where sacrifices led to a loss or draw, and score adjustments based on statistical analysis of similar material combinations
- **Code Refinements:** Minor refactorings to improve code clarity and maintainability

## Notes

Due to the lack of an ARM-based (Apple Silicon) computer, the "apple-silicon" builds are untested.

## Installation
The chess engine is available for Windows and Linux and requires a 64 Bit CPU.
There are optimized executables available for different CPU micro-architecture generations.

Starting with Velvet v4.1.0 there are also builds for macOS provided.
Currently there are no specific optimizations for the ARM-based/Apple Silicon builds implemented, so
the macOS builds for x86_64 might be faster.

If you have a relatively modern CPU (2013+) with AVX2 support, then the *...-x86_64-avx2* executable is highly recommended for best performance.

| Executable         | Description                                                  | Min. CPU Generation           | Required Instruction Sets |
|--------------------|--------------------------------------------------------------|-------------------------------| ------------------------- |
| x86_64-avx2        | Recommended for best performance on a modern CPU             | Intel Haswell / Zen1          | AVX2, BMI1                |
| x86_64-sse4-popcnt | Lower performance, recommended for CPUs without AVX2 support | Intel Nehalem / AMD Bulldozer | SSE4.2, SSE3, POPCNT      |
| x86_64-nopopcnt    | Lowest performance, but compatible with most x86_64 CPUs     | ---                           | SSE2, CMOV                |
| apple-silicon      | Native builds for Apple Silicon processors (ARM aarch64)     | Apple M1                      |                           |
| aarch64-rpi        | Linux aarch64 / for Raspberry Pi with 64-bit support         | ARM Cortex-A53                |                           |
