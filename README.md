# Supply Chain Greenwashing Evolution

## Description
Reproducibility materials for the manuscript:
"Conditional Effects of Network Position,
Payoff Uncertainty, and Strategic Inertia
on Supply Chain Greenwashing Evolution"

## Requirements
Python 3.x
NetworkX
NumPy
SciPy
Matplotlib
Pandas

## Experiments
Experiment 1: Initial Hub strategies
Experiment 2: Payoff uncertainty
Experiment 3: Payoff boundary sensitivity
Experiment 4: Learning intensity and inertia
Experiment 5: Network structure and ablation
Experiment 6: Initial greenwashing proportion

## Reproducibility
Root random seed: 42
Baseline network: N=100, m=3
Simulation horizon: 1000 steps

Experiments 2–6:
20 independent networks
20 Monte Carlo replicates per network

## Statistical Analysis
Network-level Student's t confidence intervals.
Friedman tests and paired Wilcoxon tests
with Holm adjustment.

## Data
The results directory contains simulation
outputs and statistical summaries.
