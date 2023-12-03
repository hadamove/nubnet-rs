# Source code

## Building and running

- Requires [Rust toolchain](https://www.rust-lang.org/tools/install) 1.74.0 or later
- `cargo build` to build the project
- `cargo run` to build and run the project
- *Tip*: Use `--release` flag for better performance

## Project overview

The project is structured into two separate crates:

- A library crate (`src/lib.rs`) containing general code for neural network, that can be reused in other projects.
- A binary crate (`src/bin/train_mnist.rs`) that is specific to the MNIST dataset and uses the library crate.

## Rermarks

- We tried experimenting with functional approach, although it does not always combine well with immutable data structures and Rust's ownership model
- The main focus was speed, a single epoch (going through all training examples) on an M1 Macbook Air takes about 2 seconds
- Multi-threading helps a lot, but strangely only up to 32 threads, after which it gets slower again (probably due to communication overhead)
