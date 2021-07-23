## :chess_pawn: Velvet Chess Engine

![Release](https://img.shields.io/github/v/release/mhonert/velvet-chess)
![Test](https://img.shields.io/github/workflow/status/mhonert/velvet-chess/Test?label=Test&logo=github)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**Velvet Chess Engine** is a UCI chess engine written in [Rust](https://www.rust-lang.org).

Version 1.2.0 is currently ranked around 2750 ELO in the Computer Chess Rating Lists (CCRL):
- [CCRL Blitz](https://www.computerchess.org.uk/ccrl/404/cgi/compare_engines.cgi?family=Velvet&print=Rating+list)
- [CCRL 40/15](https://www.computerchess.org.uk/ccrl/4040/cgi/compare_engines.cgi?family=Velvet&print=Rating+list)

In order to play against Velvet, you need a Chess GUI with support for the UCI protocol.
The engine was tested with **cutechess-cli** and **PyChess** on Linux and **Arena** and **Banksia** on Windows, but
should also work with other UCI compatible clients.

### :inbox_tray: Download

Executables for Windows and Linux can be downloaded from the [releases page](https://github.com/mhonert/velvet-chess/releases).

### :computer: Manual compilation

Since Velvet is written in Rust, a manual compilation requires the installation of the Rust tool chain (e.g. using [rustup](https://rustup.rs/)).
The engine currently requires a nightly build in order to use some SIMD optimizations.

```shell
rustup override set nightly
```

Then you can compile the engine using **cargo**:

```shell
cargo build --release --bin velvet
```

### :scroll: License
This project is licensed under the GNU General Public License - see the [LICENSE](LICENSE) for details.

### :tada: Attributions
- The [Chess Programming Wiki (CPW)](https://www.chessprogramming.org/Main_Page) has excellent articles and descriptions
- The testers from the [Computer Chess Rating Lists (CCRL)](https://www.computerchess.org.uk/ccrl/) are doing a great job testing lots
  of chess engines for the rating lists
