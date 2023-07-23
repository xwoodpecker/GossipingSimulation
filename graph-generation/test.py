import math

import networkx
import networkx as nx
import random

def powerlaw_cluster_graph(n, m, p, seed=None):
    """Holme and Kim algorithm for growing graphs with powerlaw
    degree distribution and approximate average clustering.

    Parameters
    ----------
    n : int
        the number of nodes
    m : float
        the number of random edges to add for each new node (can be a float now)
    p : float,
        Probability of adding a triangle after adding a random edge
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Notes
    -----
    (The notes and references remain unchanged)

    """

    if m < 1 or n < m:
        raise nx.NetworkXError(f"NetworkXError must have m>1 and m<n, m={m},n={n}")

    if p > 1 or p < 0:
        raise nx.NetworkXError(f"NetworkXError p must be in [0,1], p={p}")

    m1 = int(m)
    m2 = m1 + 1
    p_float = 1- (m - m1)

    def choose_m():
        # Pick which m to use (m1 or m2)
        if seed.random() < p_float:
            return m1
        else:
            return m2

    # Pick which m to use (m1 or m2)
    m = choose_m()
    G = nx.empty_graph(m)  # add int(m) initial nodes
    repeated_nodes = list(G.nodes())  # list of existing nodes to sample from
    source = m  # next node is m
    while source < n:  # Now add the other n-1 nodes
        possible_targets = _random_subset(repeated_nodes, m, seed)
        # do one preferential attachment for the new node
        target = possible_targets.pop()
        G.add_edge(source, target)
        repeated_nodes.append(target)  # add one node to the list for each new link
        count = 1
        while count < m:  # add int(m)-1 more new links
            if seed.random() < p:  # clustering step: add a triangle
                neighborhood = [
                    nbr
                    for nbr in G.neighbors(target)
                    if not G.has_edge(source, nbr) and not nbr == source
                ]
                if neighborhood:  # if there is a neighbor without a link
                    nbr = seed.choice(neighborhood)
                    G.add_edge(source, nbr)  # add triangle
                    repeated_nodes.append(nbr)
                    count += 1
                    continue  # go to the top of the while loop
            # else do preferential attachment step if the above fails
            target = possible_targets.pop()
            G.add_edge(source, target)
            repeated_nodes.append(target)
            count += 1

        repeated_nodes.extend([source] * m)  # add the source node to the list int(m) times
        source += 1
        # Pick which m to use (m1 or m2)
        m = choose_m()
    return G

# Helper function
def _random_subset(seq, m, seed):
    """ Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Note: Method derived from the NetworkX library.
    """
    targets = set()
    while len(targets) < m:
        x = seed.choice(seq)
        targets.add(x)
    return targets

# Example usage:
while True:
    graph2 = networkx.powerlaw_cluster_graph(n=100, m=2, p=0.3, seed=random)
    graph2_me = powerlaw_cluster_graph(n=100, m=2.0, p=0.3, seed=random)
    graph225 = powerlaw_cluster_graph(n=100, m=2.25, p=0.3, seed=random)
    graph25 = powerlaw_cluster_graph(n=100, m=2.5, p=0.3, seed=random)
    graph275 = powerlaw_cluster_graph(n=100, m=2.75, p=0.3, seed=random)
    graph3_me = powerlaw_cluster_graph(n=100, m=3.0, p=0.3, seed=random)
    graph3 = networkx.powerlaw_cluster_graph(n=100, m=3, p=0.3, seed=random)
    print()
