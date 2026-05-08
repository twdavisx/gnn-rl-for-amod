## Fork Changes

This is a fork of [DanieleGammelli/gnn-rl-for-amod](https://github.com/DanieleGammelli/gnn-rl-for-amod).

### Removed: IBM CPLEX

The original repo used CPLEX for two optimization problems:

- **Passenger matching** — solved as a linear program to maximize revenue
- **Vehicle rebalancing** — solved as a minimum-cost flow LP


### Replacement: Greedy FIFO Matching

Passenger-to-vehicle matching now uses a greedy FIFO algorithm:

- At each timestep, demand requests are shuffled to simulate random arrival order
- Each request is served if a vehicle is available in the origin zone, otherwise dropped
- more realistic

### Replacement: NetworkX Min-Cost Flow (Rebalancing)

Vehicle rebalancing uses `networkx.min_cost_flow`:

- Pure Python, no subprocess calls or file I/O
- tests were created to assure equivalent solution quality to CPLEX
