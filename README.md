# Decision Trees from Scratch

This project is a clean, from-scratch implementation of decision tree algorithms
in Clojure. It was written as a self-teaching exercise to learn the low-level
details of how decision trees work by building them from first principles.

## Overview

The primary goal of this repository is to demonstrate how decision trees
evaluate splits, calculate impurity reductions, and recursively construct tree
structures without relying on external machine learning libraries. While it
currently features a CART (Classification and Regression Trees) implementation,
the codebase is intentionally modular to support the addition of various other
tree-based algorithms in the future. It supports both categorical and continuous
features, as well as multiple loss functions for different types of target
variables.

## Code Structure

The source code is organized into distinct logical components to isolate
responsibilities:

- `src/trees_from_scratch/dataset.clj`: Manages dataset representation as a
  composite map of columns and type metadata. Contains the core logic for
  partitioning datasets based on continuous and categorical splits.
- `src/trees_from_scratch/loss.clj`: Implements various impurity metrics and
  loss functions, including Gini impurity, Shannon entropy, Mean Squared
  Deviation (MSE), and Poisson deviance.
- `src/trees_from_scratch/maths.clj`: Provides foundational mathematical
  utilities required by the loss and tree-building functions.
- `src/trees_from_scratch/stopping.clj`: Defines strategies for halting tree
  growth, providing mechanisms for both early exits (e.g., maximum depth,
  minimum sample sizes) and late exits (insufficient loss reduction).
- `src/trees_from_scratch/trees/binary.clj`: Defines the data structure of the
  resulting models (leaf nodes and split nodes) and contains the logic for
  traversing the tree to make predictions on new data rows.
- `src/trees_from_scratch/models/cart.clj`: The core implementation of the CART
  algorithm. Evaluates potential splits across all features, calculates loss
  reductions, and orchestrates the recursive training loop.
- `src/trees_from_scratch/models/core.clj`: Exposes higher-level functions for
  making predictions across entire datasets and evaluating model accuracy or
  error.

## Development Environment

This project uses `devenv` (via `devenv.nix`) to manage the Clojure toolchain
and provide convenient development scripts.

To enter the environment, run:

```bash
devenv shell
```

Once inside the shell, you can use the following built-in commands:

- `test` (or `run-tests`): Runs the Clojure test suite.
- `fmt`: Formats all source and test files using `cljfmt`.
- `lint`: Lints the codebase using `clj-kondo`.
- `check`: Runs formatting, linting, and tests in sequence.

## License

This project is licensed under the **GNU Lesser General Public License v3.0**
(LGPLv3). See the [LICENSE.md](LICENSE.md) file for details. This allows anyone
to learn from, use, and modify the project, while ensuring that the core library
remains open-source.
