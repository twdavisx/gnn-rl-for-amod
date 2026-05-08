import networkx as nx
from collections import defaultdict


def solveRebFlow(env, desiredAcc):
    """Rebalancing solver using NetworkX min-cost flow."""
    t = env.time
    flow = defaultdict(float)

    current = {n: int(env.acc[n][t+1]) for n in env.region}
    desired = {n: int(round(desiredAcc[n])) for n in env.region}

    supply = {n: current[n] - desired[n] for n in env.region}
    total_surplus = sum(v for v in supply.values() if v > 0)
    total_deficit = sum(-v for v in supply.values() if v < 0)
    balance = total_surplus - total_deficit

    G = nx.DiGraph()
    for n in env.region:
        G.add_node(n, demand=-supply[n])
    for i, j in env.G.edges:
        G.add_edge(i, j, weight=env.G.edges[i, j]['time'], capacity=current[i])

    if balance > 0:
        G.add_node('sink', demand=balance)
        for n in env.region:
            G.add_edge(n, 'sink', weight=0, capacity=balance)
    elif balance < 0:
        G.add_node('source', demand=balance)
        for n in env.region:
            G.add_edge('source', n, weight=0, capacity=-balance)

    try:
        flow_dict = nx.min_cost_flow(G)
        for i in env.region:
            for j in env.region:
                if j in flow_dict.get(i, {}) and flow_dict[i][j] > 0:
                    flow[i, j] = flow_dict[i][j]
    except nx.NetworkXUnfeasible:
        pass

    return [flow[i, j] for i, j in env.edges]
