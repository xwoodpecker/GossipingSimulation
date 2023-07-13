from itertools import combinations, groupby

import community
import networkx as nx

def apply_louvain(graph):
    """
       Apply Louvain algorithm for community detection on the given graph.
       Sets the attribute 'louvain_community'.

       Args:
           graph (NetworkX graph): The graph to apply the algorithm to.

       Returns:
           int: The number of communities
           dict: The partition, with communities numbered from 0 to number of communities
       """
    # apply the Louvain method to detect communities
    partition = community.best_partition(graph)

    # mark the communities in the graph
    for node, community_id in partition.items():
        graph.nodes[node]['community'] = community_id

    return max(partition.values()) + 1, partition


def compute_modularity(graph):
    """
      Compute the modularity of the given graph.

      Args:
          graph (NetworkX graph): The graph to compute the modularity of.

      Returns:
          float: The modularity of the graph.
    """

    communities = [set(node for node, attr in graph.nodes(data=True) if attr['community'] == c)
                   for c in set(nx.get_node_attributes(graph, 'community').values())]

    # Compute the modularity
    return nx.algorithms.community.modularity(graph, communities)

import random
import networkx as nx

def generate_simple_graph(node_count, comm_count, num_intra, num_inter, equal_sized):
    """
    Generate a modular graph with the specified parameters.

    Args:
        node_count (int): The total number of nodes in the graph.
        comm_count (int): The number of communities in the graph.
        equal_sized (bool): Whether the communities should be of equal size.
        num_intra (int): The number of intra-community edges.
        num_inter (int): The number of inter-community edges.

    Returns:
        A NetworkX graph representing the generated modular graph.
    """
    # Create an empty graph
    graph = nx.Graph()

    # Generate a list of community sizes
    sizes = []
    remaining_nodes = node_count
    if equal_sized:
        sizes = [node_count // comm_count] * comm_count
        sizes[0] += node_count % comm_count
    else:
        for i in range(comm_count - 1):
            comm_size = random.randint(1, remaining_nodes - (comm_count - i - 1))
            sizes.append(comm_size)
            remaining_nodes -= comm_size
        sizes.append(remaining_nodes)

    # Generate a random modular graph
    for i in range(comm_count):
        nodes = list(range(sum(sizes[:i]), sum(sizes[:i + 1])))
        size = sizes[i]
        subgraph = nx.Graph()
        subgraph.add_nodes_from(nodes)
        node_range = range(nodes[0], nodes[size - 1] + 1)
        dim = 2
        edges = combinations(node_range, dim)

        num_edges = 0
        for _, node_edges in groupby(edges, key=lambda x: x[0]):
            node_edges = list(node_edges)
            random_edge = random.choice(node_edges)
            subgraph.add_edge(*random_edge)
            num_edges += 1

        def random_combination(iterable, r):
            pool = tuple(iterable)
            n = len(pool)
            indices = sorted(random.sample(range(n), r))
            return tuple(pool[i] for i in indices)

        while num_edges < num_intra / comm_count:
            random_edge = random_combination(node_range, dim)
            if random_edge not in subgraph.edges:
                subgraph.add_edge(*random_edge)
                num_edges += 1

        graph.add_nodes_from(subgraph.nodes())
        graph.add_edges_from(subgraph.edges())

    # Set the community dictionary
    community_dict = {}
    for i in range(comm_count):
        nodes = list(range(sum(sizes[:i]), sum(sizes[:i + 1])))
        for node in nodes:
            community_dict[node] = i

    # Set the community attribute for each node in the graph
    nx.set_node_attributes(graph, community_dict, 'desired_community')

    if comm_count > 1:
        # Generate random numbers for each pair of tuples
        pair_numbers = [random.randint(0, num_inter) for _ in range(comm_count * (comm_count - 1) // 2)]
        total_numbers = sum(pair_numbers)

        # Normalize the numbers to ensure the sum is equal to num_inter
        normalized_numbers = [int(num * num_inter / total_numbers) for num in pair_numbers]
        remaining = num_inter - sum(normalized_numbers)

        # Distribute the remaining difference to the first few pairs
        for i in range(remaining):
            normalized_numbers[i] += 1
    else:
        normalized_numbers = [num_inter]

    index = 0
    # Add additional edges between communities based on the assigned numbers
    for i in range(comm_count):
        for j in range(i + 1, comm_count):
            num_edges = normalized_numbers[index]
            index += 1
            add_inter_community_edges(graph, i, j, num_edges)

    # Make sure that the whole graph is interconnected
    interconnected_graph = make_fully_interconnected(graph)

    return interconnected_graph


def add_inter_community_edges(graph, comm1, comm2, num_edges):
    """
    Add a specified number of inter-community edges between two communities.

    Args:
        graph (NetworkX graph): The graph to add edges to.
        comm1 (int): Community 1 index.
        comm2 (int): Community 2 index.
        num_edges (int): The number of edges to add between the communities.
    """
    nodes1 = [node for node, attr in graph.nodes(data=True) if attr['desired_community'] == comm1]
    nodes2 = [node for node, attr in graph.nodes(data=True) if attr['desired_community'] == comm2]
    edges = []

    while len(edges) < num_edges:
        node1 = random.choice(nodes1)
        node2 = random.choice(nodes2)
        edge = (node1, node2)

        if edge not in edges and not graph.has_edge(*edge):
            edges.append(edge)

    graph.add_edges_from(edges)


def make_fully_interconnected(graph):
    """
    Make sure that the entire graph is interconnected.

    Args:
        graph (NetworkX graph): The graph to interconnect.

    Returns:
        The interconnected graph.
    """
    connected_components = list(nx.connected_components(graph))
    num_components = len(connected_components)

    if num_components > 1:
        for i in range(num_components - 1):
            component1 = connected_components[i]
            component2 = connected_components[i + 1]
            node1 = random.choice(list(component1))
            node2 = random.choice(list(component2))
            graph.add_edge(node1, node2)

    return graph

#G = nx.watts_strogatz_graph(1000, 2, 0.3)
#G = nx.spectral_graph_forge(G, 0, transformation='modularity')

G = generate_simple_graph(1000,1,5000,0,True)

# compute louvain communities
num_communities, partition = apply_louvain(G)
# compute actual modularity based on the louvain communities
computed_modularity = compute_modularity(G)

print(computed_modularity)