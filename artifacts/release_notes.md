
This release mainly focussed on search-related performance optimizations and improvements :zap: 
> It is based upon my previous web-based engine Wasabi, which can be played [**here**](https://mhonert.github.io/chess).

## Changes
- Optimized move generation and perform/undo move logic
- Replaced null-move pruning with a null-move reduction approach
- Reduced transposition table entry size from 12 to 8 bytes
- Refactored move generation and move sorting to reduce memory allocation overhead 
- Sort quiet moves incrementally
- Improve history heuristic tables
- Reduce search depth for remaining moves, after a recapture move
- Null move: reduce more for higher remaining search depths 
- Move best root move to the top of the move list instead of sorting by scores
- Fix relative mate scores in transposition table
- Optimize margins for futile move reductions
- UCI support
  - Output seldepth info
  - support 'nodes' parameter in 'go' command

## Installation
- Download the suitable executable for your platform (Linux or Windows) and CPU generation
  - *x86_64-modern* - recommended for recent CPUs from 2013 onwards (requires a CPU with support for the [BMI1](https://en.wikipedia.org/wiki/Bit_Manipulation_Instruction_Sets) instruction sets)
  - *x86_64-popcnt* - for older 64-Bit CPUs, which support the POPCNT instruction, but not BMI1
  - *x86_64-vintage* - for older 64-Bit CPUs, which support neither POPCNT nor BMI1
 