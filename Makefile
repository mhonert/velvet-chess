RUSTFLAGS ?= -Ctarget-cpu=native
CFLAGS ?= -march=native
CARGO_OPTS ?= --bin velvet --features=fathomrs

export RUSTFLAGS
export CFLAGS
export CARGO_OPTS

default: build

# Regular build
build:
	cargo build --release $(CARGO_OPTS)

# Profile Guided Optimization build
pgo-build: pgo-profile pgo-optimize

# Profile Guided Optimization: create profile
pgo-profile:
	cargo pgo run -- $(CARGO_OPTS) -- multibench

# Profile Guided Optimization: create optimized build
pgo-optimize:
	cargo pgo optimize build -- $(CARGO_OPTS)

# Install cargo sub command for Profile Guided Optimization
pgo-init:
	cargo install cargo-pgo
	rustup component add llvm-tools
	cargo pgo info || echo "... but BOLT is not used here, so it is OK, if [llvm-bolt] and [merge-fdata] are not available."

# Run tests
test:
	cargo test --no-default-features --release -p velvet --lib

# Show current build configuration
show-config:
	@echo "RUSTFLAGS: $(RUSTFLAGS)"
	@echo "CARGO_OPTS: $(CARGO_OPTS)"
	@echo "CFLAGS: $(CFLAGS)"

.PHONY: show-config pgo-init test pgo-build build default pgo-optimize pgo-profile