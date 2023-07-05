import math
import random

import networkx as nx
from community import modularity, community_louvain
from networkx.algorithms.community import label_propagation_communities
import community
import numpy as np

def generate_sbm_graph(n, c, p1, p2):
    # Compute community sizes
    community_sizes = [n // c] * c
    remainder = n % c
    for i in range(remainder):
        community_sizes[i] += 1

    # Compute community probabilities
    community_probs = [[p1 if i == j else p2 for j in range(c)] for i in range(c)]

    # Generate SBM graph
    G = nx.stochastic_block_model(community_sizes, community_probs)
    return G


def generate_simple_graph(node_count, comm_count, p_intra, p_inter, equal_sized):
    """
    Generate a modular graph with the specified parameters.

    Args:
        node_count (int): The total number of nodes in the graph.
        comm_count (int): The number of communities in the graph.
        equal_sized (bool): Whether the communities should be of equal size.
        p_intra (float): The probability of intra-community edges (0 to 1).
        p_inter (float): The probability of inter-community edges (0 to 1).

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
        subgraph = nx.gnp_random_graph(size, p=p_intra)
        mapping = dict(zip(range(size), nodes))
        subgraph = nx.relabel_nodes(subgraph, mapping)
        graph.add_nodes_from(subgraph.nodes())
        graph.add_edges_from(subgraph.edges())

    # set the community dictionary
    community_dict = {}
    for i in range(comm_count):
        nodes = list(range(sum(sizes[:i]), sum(sizes[:i + 1])))
        for node in nodes:
            community_dict[node] = i

    # Set the community attribute for each node in the graph
    nx.set_node_attributes(graph, community_dict, 'desired_community')

    # Add additional edges between communities with probability p
    for i in range(comm_count):
        for j in range(i + 1, comm_count):
            add_inter_community_edges(graph, i, j, p_inter)

    # make sure that the whole graph is interconnected
    interconnected_graph = make_fully_interconnected(graph)

    return interconnected_graph


def add_inter_community_edges(graph, i, j, p):
    """
    Add inter-community edges between two communities i and j in the graph.

    Args:
        graph (NetworkX graph): The graph to add edges to.
        i (int): The ID of the first community.
        j (int): The ID of the second community.
        p (float): The probability of adding an edge between any two nodes in the two communities.

    Returns:
        A NetworkX graph containing the newly added edges
    """
    # Find nodes in communities i and j
    nodes_i = [n for n in graph.nodes if graph.nodes[n]['desired_community'] == i]
    nodes_j = [n for n in graph.nodes if graph.nodes[n]['desired_community'] == j]
    for node_i in nodes_i:
        for node_j in nodes_j:
            if random.uniform(0, 1) < p:
                graph.add_edge(node_i, node_j)
    return graph


def add_random_inter_community_edge(graph, i, j):
    """
    Add a random inter-community edge between two communities i and j in the graph.

    Args:
        graph (NetworkX graph): The graph to add an edge to.
        i (int): The ID of the first community.
        j (int): The ID of the second community.

    Returns:
        A NetworkX graph containing the newly added edges
    """
    # Find nodes in communities i and j
    nodes_i = [n for n in graph.nodes if graph.nodes[n]['desired_community'] == i]
    nodes_j = [n for n in graph.nodes if graph.nodes[n]['desired_community'] == j]
    node_i = random.choice(nodes_i)
    node_j = random.choice(nodes_j)
    graph.add_edge(node_i, node_j)
    return graph


def make_fully_interconnected(graph):
    """
    Make the graph fully interconnected by adding edges between components.

    Args:
        graph (NetworkX graph): The graph to make fully interconnected.

    Returns:
        NetworkX graph: The fully interconnected graph.
    """
    # components are non-connected graph partitions
    components = list(nx.connected_components(graph))
    num_components = len(components)

    # Stop if the graph is already fully interconnected
    if num_components < 2:
        # No need to add edges if the graph is already fully interconnected
        return graph

        # Randomly select two components to connect
    component1 = random.choice(components)
    component2 = random.choice(components)

    while component1 == component2:
        component2 = random.choice(components)

    # Randomly select one node from each component
    node1 = random.choice(list(component1))
    node2 = random.choice(list(component2))

    # Create a new graph and add edges from the existing graph
    new_graph = nx.Graph(graph)
    new_graph.add_edge(node1, node2)

    # recursive call of make_fully_interconnected
    return make_fully_interconnected(new_graph)

def generate_low_modularity_graph(num_nodes, num_edges):
    # Create an empty graph
    graph = nx.Graph()

    # Add nodes to the graph
    graph.add_nodes_from(range(num_nodes))

    # Add edges to the graph randomly
    while graph.number_of_edges() < num_edges:
        # Choose two random nodes
        node1 = random.randint(0, num_nodes - 1)
        node2 = random.randint(0, num_nodes - 1)

        # Add an edge between the chosen nodes if it doesn't already exist
        if node1 != node2 and not graph.has_edge(node1, node2):
            graph.add_edge(node1, node2)

    return graph

def generate_low_modularity_graph(num_nodes, num_edges):
    # Generate a degree sequence with low modularity
    degree_sequence = [2] * num_nodes  # Start with a uniform degree sequence

    # Add extra edges to lower the modularity
    extra_edges = num_edges - (num_nodes * 2)  # Subtract the edges from the initial uniform degree sequence
    for _ in range(extra_edges):
        node = random.randint(0, num_nodes - 1)
        degree_sequence[node] += 1

    # Create a graph with the specified degree sequence
    graph = nx.configuration_model(degree_sequence)

    # Remove self-loops and parallel edges
    graph = nx.Graph(graph)

    return graph

def generate_scale_free_graph(node_count, alpha, beta, gamma, N):
    # Generate a scale-free graph with the specified number of nodes and parameters.
    combined_graph = nx.Graph()
    iterations = math.ceil(N)
    for i in range(iterations):
        graph = nx.scale_free_graph(node_count, alpha=alpha, beta=beta, gamma=gamma)
        graph = nx.DiGraph(graph)
        if N - i >= 1:
            sample_mult = 1
        else:
            sample_mult = N % 1
        all_edges = list(graph.edges())
        sample_size = int(sample_mult * len(all_edges))
        sample_edges = random.sample(all_edges, sample_size)
        combined_graph.add_edges_from(sample_edges)

    combined_graph.remove_edges_from(nx.selfloop_edges(combined_graph))
    graph_undirected = nx.to_undirected(combined_graph)
    fully_interconnected_graph = make_fully_interconnected(graph_undirected)
    return fully_interconnected_graph


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


def generate_graph_with_low_modularity(num_nodes, num_edges):
    # Create a complete graph
    complete_graph = nx.complete_graph(num_nodes)

    # Get all possible edges
    all_edges = list(complete_graph.edges())

    # Randomly select edges to retain
    edges_to_retain = random.sample(all_edges, num_edges)

    # Create a new graph by removing non-selected edges
    graph_with_low_modularity = nx.Graph()
    graph_with_low_modularity.add_nodes_from(complete_graph.nodes())
    graph_with_low_modularity.add_edges_from(edges_to_retain)

    return graph_with_low_modularity

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

n = 1000
c = 100
p1 = 0
p2 = 0.005

# Generate a graph using the Stochastic Block Model
# G = generate_sbm_graph(n, c, p1, p2)
# G = nx.watts_strogatz_graph(100, 3, 0.1)
# G = nx.erdos_renyi_graph(100, 0.01)
# G = generate_scale_free_graph(n, 0.01, 0.98, 0.01, 20)
# G = nx.complete_graph(1000)
# G = generate_graph_with_low_modularity(1000, 35000) <--- WINNAHHHHH!!!

G = generate_scale_free_graph(n, 0.05, 0.85, 0.1, 25) # even better than the wiinnahhh


# Example usage
alpha = 2.5  # Power-law exponent
desired_modularity = 0.3  # Desired modularity


partition = community_louvain.best_partition(G)
#computed_modularity = modularity(partition, G)


# compute louvain communities
num_communities, partition = apply_louvain(G)

# compute actual modularity based on the louvain communities
computed_modularity = compute_modularity(G)

# If the modularity is greater than 0.1, regenerate the graph until it's below 0.1
while computed_modularity >= 0.1:
    print(f'fail {computed_modularity}')

    # G = generate_simple_graph(1000, 20, 0.001, 0.015, False)
    # G = generate_graph_with_low_modularity(1000, 35000) <---WINNAAAAAAAAAAH!
    G = generate_scale_free_graph(n, 0.05, 0.85, 0.1, 25)
    partition = community_louvain.best_partition(G)
    #computed_modularity = modularity(partition, G)


    # compute louvain communities
    num_communities, partition = apply_louvain(G)

    # compute actual modularity based on the louvain communities
    computed_modularity = compute_modularity(G)

print("Graph with modularity <", 0.1, "generated successfully!")
print("Modularity:", computed_modularity)
