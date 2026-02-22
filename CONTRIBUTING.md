# Contributing to ALICE-Synth

## Build

```bash
cargo build
cargo build --no-default-features   # no_std check (requires alloc)
```

## Test

```bash
cargo test
```

## Lint

```bash
cargo clippy -- -W clippy::all
cargo fmt -- --check
cargo doc --no-deps 2>&1 | grep warning
```

## Design Constraints

- **no_std core**: all modules must compile without `std`. Use `alloc::vec::Vec` behind `#[cfg]`.
- **Reciprocal constants**: pre-compute `1.0 / N` as `const` to avoid per-sample division.
- **Phase accumulator**: oscillators use 0.0..1.0 phase, not radians. Keeps math uniform.
- **Compact patches**: instrument timbres encoded as fixed-size parameter structs, not sample data.
- **Score format**: 4-byte note events with bit-packed fields. Header carries tempo and tick division.
