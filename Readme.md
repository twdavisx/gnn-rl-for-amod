# The Regime-Dependent Cost of Equity in Autonomous Mobility-on-Demand Rebalancing

**Accepted for presentation at the 2027 Transportation Research Board (TRB) Annual Meeting.**  
**Preprint:** [Read the paper](PREPRINT_URL)

This repository contains the code used for the paper
*The Regime-Dependent Cost of Equity in Autonomous Mobility-on-Demand
Rebalancing*. It includes data-generation scripts, validation of the Poisson
demand assumption, GNN-A2C model and reward implementations, the training
and evaluation code used in the experiments, as well as results.

This repository is a fork of
[DanieleGammelli/gnn-rl-for-amod](https://github.com/DanieleGammelli/gnn-rl-for-amod).
The original framework was adapted to study the tradeoffs among operator
profit, system efficiency, and spatial service equity under different
fleet-supply and demand conditions.

## Fork Changes

### Removal of IBM CPLEX

The original repository used IBM CPLEX for two optimization problems:

- **Passenger matching:** a linear program that assigns requests to vehicles
  to maximize revenue
- **Vehicle rebalancing:** a minimum-cost flow problem that determines how
  idle vehicles should be repositioned

This fork removes the IBM CPLEX dependency and replaces both components with
open-source implementations.

### Randomized FCFS Passenger Matching

Passenger matching now uses a greedy first-come-first-served procedure:

- At each timestep, newly arriving requests are shuffled uniformly at random
  to determine their within-timestep processing order.
- A request is served if an idle vehicle is available in its origin zone when
  the request is processed.
- Otherwise, the request is dropped and is not retained in a waiting queue.

This matching procedure is used consistently across all experimental
conditions.

### NetworkX Minimum-Cost Rebalancing

Vehicle rebalancing is implemented using `networkx.min_cost_flow`:

- The implementation is open source and written entirely in Python.
- It does not require subprocess calls, temporary files, or commercial
  optimization software.
- Tests verify feasibility and solution quality relative to the original
  CPLEX-based implementation.

## Repository Contents

The repository includes the following major components:

- **Data generation:** scripts that process trip records and construct the
  origin–destination demand inputs used by the simulator
- **Poisson validation:** code that evaluates the conditional Poisson
  assumption using the Pearson dispersion statistic
- **Models:** GNN-A2C policy, value-function, reward, training, and evaluation
  implementations
- **Simulation:** passenger matching, vehicle movement, and rebalancing logic
- **Results:** results of testing the trained models   