
This is the second release of the Velvet Chess Engine :sparkles: 
> It is based upon my previous web-based engine Wasabi, which can be played [**here**](https://mhonert.github.io/chess).

## Changes
- Simplified and improved passed pawn evaluation
- Replaced null-move pruning with a null-move reduction approach
- Increased (theoretical) maximum search depth
- Replaced *classical* line attack generation with obstruction difference algorithm
- Improved and optimized tuning tool
- Re-tuned all evaluation parameters with a set of 7 million test positions from self-play games
- Bonus for rooks on (half) open files
- Improved move sorting
- Improved mobility evaluation
- Improved king safety evaluation

## Installation
- Download the suitable executable for your platform (Linux or Windows) and CPU generation
  - *x86_64-modern* - recommended for recent CPUs from 2013 onwards (requires a CPU with support for the [BMI1](https://en.wikipedia.org/wiki/Bit_Manipulation_Instruction_Sets) instruction sets)
  - *x86_64-popcnt* - for older 64-Bit CPUs, which support the POPCNT instruction, but not BMI1
  - *x86_64-vintage* - for older 64-Bit CPUs, which support neither POPCNT nor BMI1
 