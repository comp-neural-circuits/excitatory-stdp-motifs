# Heterogeneous plasticity rules generate diverse structures in recurrent networks

Code accompanying the manuscript

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

- Julia 1.9 or a compatible version (https://julialang.org/downloads/)
- The dependencies listed in `Project.toml`

## Getting started

Clone or download this repository, then open a Julia session in the repository directory.
Make sure you are in the right folder: it must contain the files `Project.toml` and `Manifest.toml`.

1. Start Julia: `julia`
2. Enter the package mode by pressing `]`
3. Activate the environment of the current folder: `activate .`
4. Install the packages: `instantiate`
   (if DrWatson is reported as missing, run `add DrWatson` first)
5. Leave the package mode with Backspace, then exit Julia: `exit()`

## Running the simulations

The scripts are located in the `scripts/` directory. To check that it works, run `test_small.jl` as follows.

From the repository folder, run:
```bash
julia --project=. --threads auto scripts/test_small.jl
```

If a `.jld2` file appears in `data/sims/`, the setup is working.

To run the full simulation code, choose the plasticity rule with the `RULE`
variable at the top of each script (see the comments inside the script), and run it. For example:

```bash
julia --project=. --threads auto scripts/sim_triplets.jl
```

**Important:** The full parameter sweep can require substantial computation. Run the short test first and do not launch the full sweep until you are ready.