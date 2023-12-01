#!/bin/bash

module add rust-1.74.0

echo "#################"
echo "    COMPILING    "
echo "#################"

cargo build --release

echo "#################"
echo "     RUNNING     "
echo "#################"

nice -n 19 ./target/release/train_mnist
