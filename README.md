# nubnet-rs

This repository contains a simple (newbie) implementation an MLP in Rust built from scratch with minimum dependencies (only `rand` crate for the library code). Originally, this project emerged as a requirement for the [PV021 course at FI MUNI](https://is.muni.cz/predmet/fi/podzim2023/PV021?lang=en). The task involved coding a neural network from the ground up to achieve an 88% accuracy on the Fashion MNIST dataset, a goal that has been successfully met.

The code has been made modular so that the network library can be used for other tasks as well.

## Project structure

- `src/` contains the library code
- `examples/` contains code of how to use the library (e.g. training on MNIST)

## Running Fashion MNIST example

1. Download the fashion MNIST dataset from [here](https://drive.google.com/file/d/1MDiem-JT0zA5a5_HqVSxKxgGZcmWzrmH) and extract it at the root of the repository (you should have `data` folder next to `src` and `examples`).
2. Run the training with `cargo run --example mnist --release`

## Remarks

- It seems that the functional programming approach is not the best fit for a lot of immutable data structures, especially when they are nested.
- The primary emphasis was on achieving speed. Notably, a single epoch of training on the Fashion MNIST dataset takes approximately 2 seconds on an M1 MacBook Air.
- Multi-threading helps a lot, but strangely only up to 32 threads, after which it gets slower again (probably due to communication overhead).
