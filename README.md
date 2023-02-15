## :chess_pawn: Velvet Chess Engine

[<img src="logo/velvet_logo_square.png" align="middle" alt="Velvet Logo">](logo/velvet_logo_square.png?raw=true)

![Release](https://img.shields.io/github/v/release/mhonert/velvet-chess)
![Test](https://img.shields.io/github/actions/workflow/status/mhonert/velvet-chess/test.yml?logo=github&branch=master&label=tests)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**Velvet Chess Engine** is a UCI chess engine written in [Rust](https://www.rust-lang.org).

Version 5.0.0 is currently ranked around 3380 Elo (Blitz) and 3290 Elo (40/15) in the Computer Chess Rating Lists (CCRL):
- [CCRL Blitz](https://www.computerchess.org.uk/ccrl/404/cgi/compare_engines.cgi?family=Velvet&print=Rating+list)
- [CCRL 40/15](https://www.computerchess.org.uk/ccrl/4040/cgi/compare_engines.cgi?family=Velvet&print=Rating+list)

In Fischer Random Chess, the single-threaded rating is around 3510 Elo:
- [CCRL 40/2 FRC](https://www.computerchess.org.uk/ccrl/404FRC)

In order to play against Velvet, you need a Chess GUI with support for the UCI protocol.

Starting with v3.3.0 Velvet also supports the Fischer Random Chess (Chess960) variant.
This requires a Chess GUI/UCI client which supports the Chess960 extension as described in the UCI specification.

v4.0.1 also added support for DFRC ("Double Fischer Random Chess").

### :inbox_tray: Download

Executables for Windows, macOS and Linux can be downloaded from the [releases page](https://github.com/mhonert/velvet-chess/releases).

### :computer: Manual compilation

Since Velvet is written in Rust, a manual compilation requires the installation of the Rust tool chain (e.g. using [rustup](https://rustup.rs/)).
The installed Rust version must support the Rust 2021 Edition (i.e. v1.56 and upwards).

Then you can compile the engine using **cargo**:

```shell
cargo build --release --bin velvet
```

To compile the engine without Syzygy tablebase support (e.g. when the target architecture is not supported by the Fathom library),
you can pass the `no-default-features` flag:

```shell
cargo build --no-default-features --release --bin velvet
```

### :scroll: License
This project is licensed under the GNU General Public License - see the [LICENSE](LICENSE) for details.

### :tada: Acknowledgements
- The [Chess Programming Wiki (CPW)](https://www.chessprogramming.org/Main_Page) has excellent articles and descriptions
- A big thanks to all the chess engine testers out there
