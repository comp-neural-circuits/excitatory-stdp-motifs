# Heterogeneous plasticity rules generate diverse structures in recurrent networks

Code accompanying the manuscript:

**Cusseddu C., Festa D., Miehl C., and Gjorgjieva J. (2026).** *Heterogeneous plasticity rules generate diverse structures in recurrent networks.*

- **Contact:** claudia.cusseddu@tum.de
- **Preprint:** [bioRxiv](https://www.biorxiv.org/content/10.1101/123456789)

---

> ⚠️ **Work in progress.** This repository is under active development and is
> updated regularly as the manuscript progresses. Scripts, file names and
> folder structure may change, and some scripts are not yet available.

---

## Overview

This repository contains Julia scripts used to simulate recurrent spiking neural networks and investigate how different synaptic plasticity rules shape network connectivity.

All the simulations are built on our previous package developed by Dylan Festa [HawkesSimulator.jl](https://github.com/comp-neural-circuits/HawkesSimulator.jl).
Please refer to its repository for detailed documentation.

## Requirements

- Julia 1.9 or a compatible version
- The dependencies listed in `Project.toml`

## Installation

Clone or download this repository, then open a Julia session in the repository directory and run:

```julia
using Pkg
Pkg.instantiate()
```

This installs the dependencies declared for the project.

## Running the simulations

The scripts are located in the `scripts/` directory.

**Important:** The full parameter sweep can require substantial computation. Run the short test first and do not launch the full sweep until you are ready.

## Reproducibility

The project environment is specified by `Project.toml` and `Manifest.toml`. Simulation outputs are generated locally.