
This is the initial release of the Velvet Chess Engine :sparkles: 
> It is based upon my previous web-based engine Wasabi, which can be played [**here**](https://mhonert.github.io/chess).

## Changes
- Replaced *classical* line attack generation with obstruction difference algorithm
- Improved passed pawn evaluation

## Installation
- Download the suitable executable for your platform (Linux or Windows) and CPU generation
  - *x86_64-modern* - recommended for recent CPUs from 2013 onwards (requires a CPU with support for the [BMI1](https://en.wikipedia.org/wiki/Bit_Manipulation_Instruction_Sets) instruction sets)
  - *x86_64-popcnt* - for older 64-Bit CPUs, which support the POPCNT instruction, but not BMI1
  - *x86_64-vintage* - for older 64-Bit CPUs, which support neither POPCNT nor BMI1
 