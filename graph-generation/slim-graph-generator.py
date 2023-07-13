import math
import os
import random
import statistics
import warnings
from collections import Counter

import click
import community
import powerlaw
import networkx as nx
import numpy as np
import yaml
from networkx import PowerIterationFailedConvergence

from cfg import *

# for powerlaw package
np.seterr(divide='ignore', invalid='ignore')
# for nx scipy interaction
warnings.filterwarnings("ignore", category=FutureWarning)


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


def compute_average_edge_degree(graph):
    """
    Compute the average edge degree of a graph.

    Parameters:
        graph (networkx.Graph): The input graph.

    Returns:
        float: The average edge degree.

    """
    degrees = dict(graph.degree())
    total_nodes = graph.number_of_nodes()

    average_edge_degree = sum(degrees.values()) / total_nodes

    return average_edge_degree


def compute_stdev_edge_degree(graph):
    """
    Compute the standard deviation edge degree of a graph.

    Parameters:
        graph (networkx.Graph): The input graph.

    Returns:
        float: The standard deviation edge degree.

    """
    degrees = dict(graph.degree())

    stdev_edge_degree = statistics.stdev(degrees)

    return stdev_edge_degree


def compute_power_law(graph):
    """
    Compute the power-law distribution parameters for the degree sequence of a graph.

    Parameters:
        graph (networkx.Graph): The input graph.

    Returns:
        float: The estimated power-law exponent.
        float: The lower bound of the power-law region.

    """
    degrees = [degree for _, degree in graph.degree()]
    fit = powerlaw.Fit(degrees, verbose=False)
    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin
    return alpha, xmin


def compute_average_path_length(graph):
    """
    Compute the average path length of a graph.

    Parameters:
        graph (networkx.Graph): The input graph.

    Returns:
        float: The average path length.

    """
    average_path_length = nx.average_shortest_path_length(graph)
    return average_path_length


def compute_cluster_coefficient(graph):
    """
    Compute the cluster coefficient of a graph.

    Parameters:
        graph (networkx.Graph): The input graph.

    Returns:
        float: The cluster coefficient.

    """
    cluster_coefficient = nx.average_clustering(graph)
    return cluster_coefficient


def save_graph_as_adj_list(graph, name):
    """
      Save the given graph as an adjacency list.

      Args:
          graph (NetworkX graph): The graph to save.
          name (str): The name of the file to save the adjacency list as.

      Returns:
          None
      """
    if name.startswith('_'):
        name = name.lstrip('_')
    directory = './generated_graphs/'
    os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
    # Save the graph as an adjacency list
    with open(f'./generated_graphs/{name}.adj-list', 'w') as f:
        for line in nx.generate_adjlist(graph):
            f.write(line + ',')


def prompt_node_count():
    """
    Prompt the user to enter the total number of nodes in the graph.

    Returns:
        int: The total number of nodes entered by the user.
    """
    return click.prompt('Enter the total number of nodes in the graph', type=int, default=1000)


def prompt_comm_count():
    """
    Prompt the user to enter the number of communities in the graph.

    Returns:
        int: The number of communities entered by the user.
    """
    return click.prompt('Enter the number of communities in the graph', type=int, default=10)


def prompt_p_intra():
    """
    Prompt the user to enter the probability of intra-community edges.

    Returns:
        float: The probability of intra-community edges entered by the user.
    """
    return click.prompt('Enter the probability of intra-community edges (0 to 1)', type=float, default=0.1)


def prompt_p_inter():
    """
    Prompt the user to enter the probability of inter-community edges.

    Returns:
        float: The probability of inter-community edges entered by the user.
    """
    return click.prompt('Enter the probability of inter-community edges (0 to 1)', type=float, default=0.001)


def prompt_simple_equal_comms():
    """
    Prompt the user to choose whether the communities should be of equal size.

    Returns:
        bool: True if the communities should be of equal size, False otherwise.
    """
    return click.confirm('Should the communities be of equal size?', default=True)


def get_params_simple_graph():
    """
    Prompt the user to enter parameters for generating a simple graph.

    Returns:
        tuple: A tuple containing the entered values for node count, community count, intra-community edge probability,
               inter-community edge probability, and whether communities should be of equal size.
    """
    node_count = prompt_node_count()
    comm_count = prompt_comm_count()
    equal_sized = prompt_simple_equal_comms()
    p_intra = prompt_p_intra()
    p_inter = prompt_p_inter()
    return node_count, comm_count, p_intra, p_inter, equal_sized


def get_graph_properties_simple_graph(node_count, comm_count, p_intra, p_inter, equal_sized):
    """
    Get the graph properties based on the provided parameters for generating a simple graph.

    Args:
        node_count (int): The total number of nodes in the graph.
        comm_count (int): The number of communities in the graph.
        p_intra (float): The probability of intra-community edges.
        p_inter (float): The probability of inter-community edges.
        equal_sized (bool): True if the communities should be of equal size, False otherwise.

    Returns:
        dict: A dictionary containing the graph properties.
    """
    return {
        'nodeCount': node_count,
        'communityCount': comm_count,
        'probabilityIntraCommunityEdge': p_intra,
        'probabilityInterCommunityEdge': p_inter,
        'equalSizedCommunities': equal_sized
    }


def get_end_params_simple_graph():
    """
    Prompt the user to enter the end parameters for generating a simple graph.

    Returns:
        tuple: A tuple containing the entered values for node count, community count, intra-community edge probability,
               and inter-community edge probability.
    """
    node_count = prompt_node_count()
    comm_count = prompt_comm_count()
    p_intra = prompt_p_intra()
    p_inter = prompt_p_inter()
    return node_count, comm_count, p_intra, p_inter


def prompt_degree():
    """
    Prompt the user to enter the graph degree.

    Returns:
        int: The entered graph degree.
    """
    return click.prompt('Enter the graph degree', type=int, default=1)


def prompt_modularity():
    """
    Prompt the user to enter the graph modularity.

    Returns:
        float: The entered graph modularity value.
    """
    return click.prompt('Enter the graph modularity (0 to 1)', type=float, default=0.8)


def prompt_degree_distribution():
    """
    Prompt the user to enter the degree distribution function.

    Returns:
        str: The entered degree distribution function name.
    """
    return click.prompt('Enter the degree distribution function',
                        type=click.Choice([REGULAR_DISTRIBUTION_NAME, POISSON_DISTRIBUTION_NAME,
                                           GEOMETRIC_DISTRIBUTION_NAME,
                                           SCALE_FREE_DISTRIBUTION_NAME]),
                        default=POISSON_DISTRIBUTION_NAME)


def prompt_community_distribution():
    """
    Prompt the user to enter the community distribution function.

    Returns:
        str: The entered community distribution function name.
    """
    return click.prompt('Enter the community distribution function',
                        type=click.Choice(
                            [REGULAR_DISTRIBUTION_NAME, POISSON_DISTRIBUTION_NAME,
                             GEOMETRIC_DISTRIBUTION_NAME, SCALE_FREE_DISTRIBUTION_NAME]),
                        default=REGULAR_DISTRIBUTION_NAME)


def prompt_alpha():
    """
    Prompt the user to enter the alpha value.

    Returns:
        float: The entered alpha value.
    """
    return click.prompt('Enter alpha (0-1)', type=float, default=0.9)


def prompt_beta():
    """
    Prompt the user to enter the beta value.

    Returns:
        float: The entered beta value.
    """
    return click.prompt('Enter beta (0-1)', type=float, default=0.05)


def prompt_gamma():
    """
    Prompt the user to enter the gamma value.

    Returns:
        float: The entered gamma value.
    """
    return click.prompt('Enter gamma (0-1)', type=float, default=0.05)


def prompt_edge_multi():
    """
    Prompt the user to enter edge multiplier value.

    Returns:
        float: The entered edge multiplier value.
    """
    return click.prompt('Enter the edge multiplier', type=float, default=1)


def get_params_scale_free_graph():
    """
    Prompt the user to enter parameters for generating a scale-free graph.

    Returns:
        tuple: A tuple containing the entered values for node count, alpha, beta, and gamma.
    """
    node_count = prompt_node_count()
    print('Now alpha, betta and gamma can be defined, their sum must be 1')
    while True:
        alpha = prompt_alpha()
        beta = prompt_beta()
        gamma = prompt_gamma()
        if alpha + beta + gamma == 1:
            break
        else:
            print('Sum is not 1, try again')
    edge_multi = prompt_edge_multi()
    return node_count, alpha, beta, gamma, edge_multi


def get_graph_properties_scale_free_graph(node_count, alpha, beta, gamma, edge_multi):
    """
    Get the graph properties based on the provided parameters for generating a scale-free graph.

    Args:
        node_count (int): The total number of nodes in the graph.
        alpha (float): The alpha value.
        beta (float): The beta value.
        gamma (float): The gamma value.
        edge_multi (float): The edge_multiplier value.

    Returns:
        dict: A dictionary containing the graph properties.
    """
    return {
        'nodeCount': node_count,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'edgeMultiplier': edge_multi
    }


def prompt_new_edges():
    """
    Prompt the user to enter the number of new edges to add for each node in the graph.

    Returns:
        float: The entered number of new edges
    """
    return click.prompt('Enter the number of edges to attach from each newly added node', type=float, default=2.0)


def prompt_new_edges_as_int():
    """
    Prompt the user to enter the number of new edges to add for each node in the graph.

    Returns:
        int: The entered number of new edges
    """
    return click.prompt('Enter the number of edges to attach from each newly added node', type=int, default=2)


def prompt_triangle_probability():
    """
    Prompt the user to enter the probability of adding a triangle after adding a random edge.

    Returns:
        float: The probability of adding a triangle after adding a random edge
    """
    return click.prompt('Enter the probability of adding a triangle after adding a random edge', type=float,
                        default=0.5)


def get_params_barabasi_albert_graph():
    """
    Prompt the user to enter parameters for generating a Barabasi-Albert graph.

    Returns:
        tuple: A tuple containing the entered values for node count and edge degree.
    """

    node_count = prompt_node_count()
    new_edges = prompt_new_edges()
    return node_count, new_edges


def get_params_holme_kim_graph():
    """
    Prompt the user to enter parameters for generating a Holme-Kim graph.

    Returns:
        tuple: A tuple containing the entered values for node count and edge degree.
    """

    node_count = prompt_node_count()
    new_edges = prompt_new_edges_as_int()
    triangle_probability = prompt_triangle_probability()
    return node_count, new_edges, triangle_probability


def get_graph_properties_barabasi_albert_graph(node_count, new_edges):
    """
    Get the graph properties based on the provided parameters for generating a Barabasi-Albert graph.

    Args:
        node_count (int): The total number of nodes in the graph.
        new_edges (float): The number of new edges for each node in the graph.

    Returns:
        dict: A dictionary containing the graph properties.
    """
    return {
        'nodeCount': node_count,
        'newEdges': new_edges
    }


def get_graph_properties_holme_kim_graph(node_count, new_edges, triangle_probability):
    """
    Get the graph properties based on the provided parameters for generating a Holme-Kim graph.

    Args:
        node_count (int): The total number of nodes in the graph.
        new_edges (int): The number of new edges for each node in the graph.
        triangle_probability (float): Probability of adding a triangle after adding a random edge

    Returns:
        dict: A dictionary containing the graph properties.
    """
    return {
        'nodeCount': node_count,
        'newEdges': new_edges,
        'triangleProbability': triangle_probability
    }


def get_get_params_func(graph_type):
    """
    Get the corresponding get_params function based on the graph type.

    Args:
        graph_type (str): The graph type.

    Returns:
        function: The get_params function for the specified graph type.
    """

    get_params_funcs = {
        GRAPH_TYPE_SIMPLE_SHORT: get_params_simple_graph,
        GRAPH_TYPE_SCALE_FREE_SHORT: get_params_scale_free_graph,
        GRAPH_TYPE_BARABASI_ALBERT_SHORT: get_params_barabasi_albert_graph,
        GRAPH_TYPE_HOLME_KIM_SHORT: get_params_holme_kim_graph,
    }
    func = get_params_funcs.get(graph_type)
    return func


def get_get_graph_properties_func(graph_type):
    """
    Get the corresponding get_graph_properties function based on the graph type.

    Args:
        graph_type (str): The graph type.

    Returns:
        function: The get_graph_properties function for the specified graph type.
    """
    get_graph_properties_funcs = {
        GRAPH_TYPE_SIMPLE_SHORT: get_graph_properties_simple_graph,
        GRAPH_TYPE_SCALE_FREE_SHORT: get_graph_properties_scale_free_graph,
        GRAPH_TYPE_BARABASI_ALBERT_SHORT: get_graph_properties_barabasi_albert_graph,
        GRAPH_TYPE_HOLME_KIM_SHORT: get_graph_properties_holme_kim_graph,
    }
    func = get_graph_properties_funcs.get(graph_type)
    return func


def get_get_end_params_func(graph_type):
    """
    Get the corresponding get_end_params function based on the graph type.

    Args:
        graph_type (str): The graph type.

    Returns:
        function: The get_end_params function for the specified graph type.
    """
    get_end_params_funcs = {
        GRAPH_TYPE_SIMPLE_SHORT: get_end_params_simple_graph,
        GRAPH_TYPE_SCALE_FREE_SHORT: get_params_scale_free_graph,
        GRAPH_TYPE_BARABASI_ALBERT_SHORT: get_params_barabasi_albert_graph,
        GRAPH_TYPE_HOLME_KIM_SHORT: get_params_holme_kim_graph,
    }
    func = get_end_params_funcs.get(graph_type)
    return func


def get_simple_graph_name(node_count, comm_count, p_intra, p_inter, equal_sized):
    """
    Generate a name for a simple graph based on the provided parameters.

    Args:
        node_count (int): The total number of nodes in the graph.
        comm_count (int): The number of communities in the graph.
        p_intra (float): The probability of an intra-community edge.
        p_inter (float): The probability of an inter-community edge.
        equal_sized (bool): Indicates if the communities are equal-sized.

    Returns:
        str: The generated graph name.
    """
    p_intra = np.round(p_intra, decimals=4)
    p_inter = np.round(p_inter, decimals=4)
    eq_str = 'eq' if equal_sized else 'ne'
    return f'{GRAPH_TYPE_SIMPLE_SHORT}-n{node_count}-c{comm_count}-{eq_str}-p1-{p_intra}-p2-{p_inter}'


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


def get_scale_free_graph_name(node_count, alpha, beta, gamma, edge_multi):
    """
    Generate a name for a scale-free graph based on the provided parameters.

    Args:
        node_count (int): The total number of nodes in the graph.
        alpha (float): The alpha parameter for generating scale-free graphs.
        beta (float): The beta parameter for generating scale-free graphs.
        gamma (float): The gamma parameter for generating scale-free graphs.
        edge_multi (float): The edge_multiplier parameter for generating scale-free graphs.

    Returns:
        str: The generated graph name.
    """
    alpha = np.round(alpha, decimals=4)
    beta = np.round(beta, decimals=4)
    gamma = np.round(gamma, decimals=4)
    edge_multi = np.round(edge_multi, decimals=4)
    return f'{GRAPH_TYPE_SCALE_FREE_SHORT}-n{node_count}-a{alpha}-b{beta}-g{gamma}-e{edge_multi}'


def generate_scale_free_graph(node_count, alpha, beta, gamma, edge_multiplier):
    """
      Generate a scale-free graph using the networkx scale_free_graph implementation with the specified parameters.

      Args:
          node_count (int): The number of nodes in the graph.
          alpha (float): The probability of adding a new node connected to an existing node with degree `k`.
          Increasing alpha results in a higher likelihood of adding new nodes to highly connected nodes,
          leading to a more clustered graph.
          beta (float): The probability of rewiring an edge in the graph.
          The beta parameter determines the probability of rewiring an edge,
          with higher values leading to more randomization of the edges and a less clustered graph.
          gamma (float): The probability of adding a new node unconnected to any other node.
          It affects the level of connectivity in the resulting graph.
          edge_multiplier (float): The multiplier for attempting to add edges to the graph.

      Returns:
          A fully-interconnected scale-free graph of `node_count` nodes,
          generated using the networkx scale_free_graph implementation
          with the specified `alpha`, `beta`, and `gamma`.
          Self-loops are also removed from the resulting graph.
      """
    # Generate a scale-free graph with the specified number of nodes and parameters.
    combined_graph = nx.Graph()
    iterations = math.ceil(edge_multiplier)
    for i in range(iterations):
        graph = nx.scale_free_graph(node_count, alpha=alpha, beta=beta, gamma=gamma)
        graph = nx.DiGraph(graph)
        if edge_multiplier - i >= 1:
            sample_mult = 1
        else:
            sample_mult = edge_multiplier % 1
        all_edges = list(graph.edges())
        sample_size = int(sample_mult * len(all_edges))
        sample_edges = random.sample(all_edges, sample_size)
        combined_graph.add_edges_from(sample_edges)

    combined_graph.remove_edges_from(nx.selfloop_edges(combined_graph))
    graph_undirected = nx.to_undirected(combined_graph)
    fully_interconnected_graph = make_fully_interconnected(graph_undirected)
    return fully_interconnected_graph


def get_barabasi_albert_graph_name(node_count, new_edges):
    """
    Generate a name for a Barabasi-Albert graph based on the provided parameters.

    Args:
        node_count (int): The total number of nodes in the graph.
        new_edges (float): The degree of each new node added during the graph construction.

    Returns:
        str: The generated graph name.
    """
    new_edges = np.round(new_edges, decimals=4)
    return f'{GRAPH_TYPE_BARABASI_ALBERT_SHORT}-n{node_count}-e{new_edges}'


def get_holme_kim_graph_name(node_count, new_edges, triangle_probability):
    """
    Generate a name for a Barabasi-Albert graph based on the provided parameters.

    Args:
        node_count (int): The total number of nodes in the graph.
        new_edges (int): The degree of each new node added during the graph construction.
        triangle_probability (float):  Probability of adding a triangle after adding a random edge.

    Returns:
        str: The generated graph name.
    """
    triangle_probability = np.round(triangle_probability, decimals=4)
    return f'{GRAPH_TYPE_HOLME_KIM_SHORT}-n{node_count}-e{new_edges}-t{triangle_probability}'


def generate_simple_barabasi_albert_graph(node_count, new_edges):
    """
       Generates a Barabasi-Albert graph with the specified number of nodes and minimum degree.

       Args:
           node_count (int): The number of nodes in the graph.
           new_edges (float): The number of edges to attach from a new node to existing nodes.

       Returns:
           A Barabasi-Albert graph of `node_count` nodes and `new_edges` node attachments.
   """
    # Generate a barabasi albert graph with the specified number of nodes and exponent.
    graph = nx.barabasi_albert_graph(node_count, new_edges)
    return graph


def generate_barabasi_albert_graph(node_count, new_edges):
    """
       Generates a Barabasi-Albert graph with the specified number of nodes and minimum degree.
       Uses the dual barabasi albert generation method to support floating point edge degrees.

       Args:
           node_count (int): The number of nodes in the graph.
           new_edges (float): The number of edges to attach from a new node to existing nodes.

       Returns:
           A Barabasi-Albert graph of `node_count` nodes and `new_edges` node attachments.
   """

    m1 = int(new_edges)  # Assign the integer part of edge_degree to m1
    m2 = m1 + 1  # Assign m1 + 1 to m2
    p = 1 - (new_edges - int(new_edges))

    graph = nx.dual_barabasi_albert_graph(node_count, m1, m2, p)
    return graph


def generate_holme_kim_graph(node_count, new_edges, triangle_probability):
    """
       Generates a Holme-Kim graph with the specified number of nodes, number of new edges and triangle probability.
       The Holme and Kim algorithm for growing graphs
       with powerlaw degree distribution and approximate average clustering is used.

       Args:
           node_count (int): The number of nodes in the graph.
           new_edges (int): The number of edges to attach from a new node to existing nodes.
           triangle_probability (float):  Probability of adding a triangle after adding a random edge.

       Returns:
           A Holme and Kim Powerlaw Cluster Graph.
   """
    graph = nx.powerlaw_cluster_graph(node_count, new_edges, triangle_probability)
    return graph


def get_creation_func(graph_type):
    """
       Returns the appropriate graph creation function based on the given graph type.

       Args:
           graph_type (str): The type of graph.

       Returns:
           func (function): The graph creation function corresponding to the graph type.
    """
    create_graph_funcs = {
        GRAPH_TYPE_SIMPLE_SHORT: generate_simple_graph,
        GRAPH_TYPE_SCALE_FREE_SHORT: generate_scale_free_graph,
        GRAPH_TYPE_BARABASI_ALBERT_SHORT: generate_barabasi_albert_graph,
        GRAPH_TYPE_HOLME_KIM_SHORT: generate_holme_kim_graph,
    }
    func = create_graph_funcs.get(graph_type)
    return func


def get_graph_name(graph_type, graph_params):
    """
       Returns the name of the graph based on the graph type and its parameters.

       Args:
           graph_type (str): The type of graph.
           graph_params (tuple): The parameters specific to the graph type.

       Returns:
           func (function): The function that generates the name of the graph based on its type and parameters.
    """
    get_graph_name_funcs = {
        GRAPH_TYPE_SIMPLE_SHORT: get_simple_graph_name,
        GRAPH_TYPE_SCALE_FREE_SHORT: get_scale_free_graph_name,
        GRAPH_TYPE_BARABASI_ALBERT_SHORT: get_barabasi_albert_graph_name,
        GRAPH_TYPE_HOLME_KIM_SHORT: get_holme_kim_graph_name,
    }
    func = get_graph_name_funcs.get(graph_type)
    return func(*graph_params)


def generate_graph_resource_yaml(name, adjacency_list, graph_type, graph_properties,
                                 series_label=None, value_list=None, ):
    """
       Generates a YAML resource definition for a graph with its associated properties and values.

       Args:
           name (str): The name of the graph resource.
           adjacency_list (list): The adjacency list representation of the graph.
           graph_type (str): The type of graph.
           graph_properties (list): The properties of the graph.
           series_label (str, optional): The label for serial simulation (default is None).
           value_list (list, optional): The corresponding values for the properties (default is None).
    """

    # Custom Dumper Class to correctly enquote strings
    class CustomDumper(yaml.Dumper):
        pass

    # just subclass the built-in str
    class QuotedString(str):
        pass

    def quoted_scalar(dumper, data):
        # a representer to force quotations on scalars
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

    # add the QuotedString custom type with a forced quotation representer to your dumper
    CustomDumper.add_representer(QuotedString, quoted_scalar)

    for key, value in graph_properties.items():
        if isinstance(value, float):
            value = round(value, 4)
        graph_properties[key] = QuotedString(value)

    # Split the adjacency list by newlines
    adjacency_list_sections = adjacency_list.split(',')

    resource_dict = {
        'apiVersion': 'gossip.io/v1',
        'kind': 'Graph',
        'metadata': {
            'name': name,
        },
        'spec': {
            'adjacencyList':
                [QuotedString(f'{section.strip()},') for section in adjacency_list_sections if section.strip()],
            'graphType': QuotedString(graph_type),
            'graphProperties': graph_properties
        }
    }

    if series_label is not None:
        resource_dict['metadata']['labels'] = {
            'series': series_label
        }

    if value_list is not None:
        resource_dict['spec']['valueList'] = QuotedString(f'{value_list}')

    return yaml.dump(resource_dict, Dumper=CustomDumper, width=float("inf"))


def save_graph_resource_yaml(content, name):
    """
        Saves the graph resource YAML content to a file.

        Args:
            content (str): The content of the YAML resource.
            name (str): The name of the graph resource.
    """
    directory = './generated_yamls/'
    os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
    # Save the graph as an adjacency list
    with open(f'./generated_yamls/{name}.yaml', 'w') as f:
        f.write(content)


def get_graph_type_long(graph_type):
    """
        Returns the long version of the graph type based on the short version.

        Args:
            graph_type (str): The short version of the graph type.

        Returns:
            str: The long version of the graph type.
    """
    long_graph_types = {
        GRAPH_TYPE_SIMPLE_SHORT: GRAPH_TYPE_SIMPLE,
        GRAPH_TYPE_SCALE_FREE_SHORT: GRAPH_TYPE_SCALE_FREE,
        GRAPH_TYPE_BARABASI_ALBERT_SHORT: GRAPH_TYPE_BARABASI_ALBERT,
        GRAPH_TYPE_HOLME_KIM_SHORT: GRAPH_TYPE_HOLME_KIM,
    }
    return long_graph_types.get(graph_type)


CLICK_COUNT_HELP_TEXT = 'The number of graphs to be created N or MxN input with M as the number of different ' \
                        'parameters and N as the number of graphs generated for each set of parameters.\n' \
                        'C/N or C/NxM specifies a selection of C graphs with an approximated target metric.'

CLICK_COUNT_PROMPT_TEXT = 'Enter the number of graphs that are to be generated.\n' \
                          'Either enter a single number N or an MxN input ' \
                          'with M as the number of different parameters ' \
                          'and N as the number of graphs generated for each set of parameters.\n' \
                          'Enter C/N or C/NxM to select C graphs from the generated graphs ' \
                          'that have values close to a target metric'

CLICK_GRAPH_TYPE_HELP_TEXT = f'The graph type (simple, scale-free, barabasi-albert or holme-kim) for the ' \
                             f'created graph '

CLICK_GRAPH_TYPE_PROMPT_TEXT = 'Choose graph type:\n' \
                               f'* [{GRAPH_TYPE_SIMPLE_SHORT}] : Simple modular graph creation ' \
                               'based on inter/intra-edge generation\n' \
                               f'* [{GRAPH_TYPE_SCALE_FREE_SHORT}] : Scale-free graph creation\n' \
                               f'* [{GRAPH_TYPE_BARABASI_ALBERT_SHORT}] : Barabási-Albert graph creation\n' \
                               f'* [{GRAPH_TYPE_HOLME_KIM_SHORT}] : Holme-Kim powerlaw cluster graph creation\n'


@click.command()
@click.option('--count',
              type=str,
              help=CLICK_COUNT_HELP_TEXT,
              prompt=CLICK_COUNT_PROMPT_TEXT)
@click.option('--graph-type',
              type=click.Choice([f'{GRAPH_TYPE_SIMPLE_SHORT}',
                                 f'{GRAPH_TYPE_SCALE_FREE_SHORT}', f'{GRAPH_TYPE_BARABASI_ALBERT_SHORT}',
                                 f'{GRAPH_TYPE_HOLME_KIM_SHORT}']),
              help=CLICK_GRAPH_TYPE_HELP_TEXT,
              prompt=CLICK_GRAPH_TYPE_PROMPT_TEXT)
def generate_graphs(count, graph_type):
    """
        Generates graphs based on the given count and graph type.

        Args:
            count (str): The count of graphs to generate.
            graph_type (str): The type of graph.

    """
    run = True
    while run:
        try:
            selectSubset = False
            subset_size = 0
            sameParams = False
            # check if a subset is specified
            if '/' in count:
                subset_size, count = count.split('/')
                subset_size = int(subset_size)
                selectSubset = True
            # check whether single, multiple or mxn graph generation has to be done
            if 'x' in count:
                M, N = count.split('x')
                N = int(N)
                M = int(M)
                # Check if N and M are positive integers
                if N > 0 and M > 0:
                    if N * M >= subset_size:
                        run = False
                    else:
                        print("Can not select more graphs than the total amount generated.")
                else:
                    print("M and N must be positive integers. Please try again.")

            else:
                N = int(count)
                # Check if N and is a positive integers
                if N > 0:
                    if N >= subset_size:
                        sameParams = True
                        run = False
                    else:
                        print("Can not select more graphs than the total amount generated.")
                else:
                    print("N must be a positive integers. Please try again.")
        except ValueError as ve:
            print("Invalid input format. Please enter valid 'NxM' values.")
        finally:
            if run:
                count = click.prompt(
                    CLICK_COUNT_PROMPT_TEXT,
                    type=str, default="5x10")

    # set the functions for the chosen graph type
    get_params_func = get_get_params_func(graph_type)
    get_graph_properties_func = get_get_graph_properties_func(graph_type)
    get_end_params_func = get_get_end_params_func(graph_type)
    create_graph_func = get_creation_func(graph_type)

    graph_properties = {}
    graphs = {}
    name_counts = {}

    # create N graphs for the given params
    # add them and their properties to the arrays
    def create_for_N(params):
        for _ in range(0, N):
            name = get_graph_name(graph_type, params)
            print(f'Creating graph {name}...')
            graph = create_graph_func(*params)
            if name not in name_counts:
                # First occurrence of the name
                name_counts[name] = 1
            else:
                # Duplicate name found
                count = name_counts[name]
                name_counts[name] += 1
                name = f'{name}_{count}'
            graph_properties[name] = get_graph_properties_func(*params)
            graphs[name] = graph

    # get the graph parameters from user input
    graph_params = get_params_func()

    # in case of MxN graph creation
    if sameParams is True:
        # this case is for a "N" graph generation
        # all graphs with the same params
        create_for_N(graph_params)

    else:
        # this case is for a "MxN" graph generation
        one_by_one = click.confirm('Do you want to provide the different parameters one by one?', default=False)
        if one_by_one:
            create_for_N(graph_params)
            # ask each set of params one by one
            for _ in range(0, M - 1):
                graph_params = get_params_func()
                create_for_N(graph_params)
        else:
            interpolate = click.confirm('Do you want to interpolate the parameters? Otherwise they will be randomly '
                                        'generated.', default=True)
            print('Specify end parameters of the last set of graphs (end of the interval).')
            # ask for the last set of params
            end_params = get_end_params_func()

            # interpolate the remaining parameters

            def interpolate_params(start_params, end_params):
                param_lists = []
                for i in range(0, len(start_params)):
                    # some parameters can not be interpolated
                    # and were therefore not prompted for the end params
                    if i < len(end_params):
                        start = start_params[i]
                        end = end_params[i]
                        values = np.linspace(start, end, M)
                        if isinstance(start, int) and isinstance(end, int):
                            values = values.astype(int)
                    else:
                        values = np.full(M, start_params[i])
                    param_lists.append(values)

                # Transpose the input array to align the i-th elements from each inner array
                transposed_params = np.transpose(np.array(param_lists, dtype='object'))
                # Create a new structure with tuples of i-th elements from each inner array
                graph_params_list = [tuple(row) for row in transposed_params]
                return graph_params_list

            def randomly_generate_params(start_params, end_params):
                param_lists = []
                for i in range(len(start_params)):
                    # some parameters can not be interpolated
                    # and were therefore not prompted for the end params
                    if i < len(end_params):
                        start = start_params[i]
                        end = end_params[i]
                        if isinstance(start, int) and isinstance(end, int):
                            values = np.random.randint(start, end + 1, M)
                        else:
                            values = np.random.uniform(start, end, M)
                    else:
                        values = np.full(M, start_params[i])
                    param_lists.append(values)

                # Transpose the input array to align the i-th elements from each inner array
                transposed_params = np.transpose(np.array(param_lists, dtype='object'))
                # Create a new structure with tuples of i-th elements from each inner array
                graph_params_list = [tuple(row) for row in transposed_params]
                return graph_params_list

            if interpolate:
                graph_params_list = interpolate_params(graph_params, end_params)
            else:
                graph_params_list = randomly_generate_params(graph_params, end_params)
                new_graph_params_list = []
                # this is a "dirty" solution for the random generation to work for scale free
                # note that the solution makes the generation not exactly match the given interval
                if graph_type == GRAPH_TYPE_SCALE_FREE_SHORT:
                    for t in graph_params_list:
                        n, a, b, c, e = t
                        total = a + b + c
                        a /= total
                        b /= total
                        c /= total
                        new_t = (n, a, b, c, e)
                        new_graph_params_list.append(new_t)
                    graph_params_list = new_graph_params_list

            # create N graphs for each set of params
            for params in graph_params_list:
                create_for_N(params)

    # visualize in case the user wants to see the graphs
    # visualize = click.confirm('Do you want to see the created graphs?', default=False)
    # export to gephi at the end in case the user wants to see the graphs
    visualize = click.confirm('Do you want to export the created graphs for visualisation?', default=False)

    all_keys_before_set = set()
    for value in graph_properties.values():
        all_keys_before_set.update(value.keys())

    def compute_metrics():
        print(f'Computing graph metrics...')
        for name, graph in graphs.items():
            print(f'Computing metrics for graph {name}...')

            num_edges = len(graph.edges())
            graph_properties[name]['numEdges'] = num_edges

            # compute louvain communities
            num_communities, partition = apply_louvain(graph)
            graph_properties[name]['numCommunities'] = num_communities

            community_sizes = Counter(partition.values())
            average_community_size = sum(community_sizes.values()) / len(community_sizes)
            graph_properties[name]['averageCommunitySize'] = average_community_size

            community_size_std = statistics.stdev(community_sizes.values())
            graph_properties[name]['stdevCommunitySize'] = community_size_std

            avg_clustering_per_community = nx.clustering(graph, partition)
            overall_average_clustering = sum(avg_clustering_per_community.values()) / len(avg_clustering_per_community)
            graph_properties[name]['overallAverageCommunityClustering'] = overall_average_clustering
            overall_stdev_clustering = statistics.stdev(avg_clustering_per_community.values())
            graph_properties[name]['overallStdevCommunityClustering'] = overall_stdev_clustering

            # compute actual modularity based on the louvain communities
            computed_modularity = compute_modularity(graph)
            graph_properties[name]['modularity'] = computed_modularity
            # print(f'The graph has a computed modularity of {computed_modularity}.')

            computed_avg_degree = compute_average_edge_degree(graph)
            # print(f'The graph has a computed average degree of {computed_avg_degree}.')
            graph_properties[name]['averageEdgeDegree'] = computed_avg_degree
            computed_stdev_degree = compute_stdev_edge_degree(graph)
            graph_properties[name]['stdevEdgeDegree'] = computed_stdev_degree

            computed_power_law_exp, lower_bound = compute_power_law(graph)
            # print(f'The graph has a computed power law of {computed_power_law_exp} with lower bound {lower_bound}.')
            graph_properties[name]['estimatedPowerLawExponent'] = computed_power_law_exp
            graph_properties[name]['lowerBoundPowerLawRegion'] = lower_bound

            computed_avg_path_length = compute_average_path_length(graph)
            # print(f'The graph has a computed avg path length of {computed_avg_path_length}.')
            graph_properties[name]['averagePathLength'] = computed_avg_path_length

            computed_clust_coefficient = compute_cluster_coefficient(graph)
            # print(f'The graph has a computed cluster coefficient of {computed_clust_coefficient}.')
            graph_properties[name]['clusterCoefficient'] = computed_clust_coefficient

            degree_centrality = nx.degree_centrality(graph)
            average_degree_centrality = sum(degree_centrality.values()) / len(degree_centrality)
            graph_properties[name]['averageDegreeCentrality'] = average_degree_centrality
            degree_centrality_std = statistics.stdev(degree_centrality.values())
            graph_properties[name]['stdevDegreeCentrality'] = degree_centrality_std

            betweenness_centrality = nx.betweenness_centrality(graph)
            average_betweenness_centrality = sum(betweenness_centrality.values()) / len(betweenness_centrality)
            graph_properties[name]['averageBetweennessCentrality'] = average_betweenness_centrality
            betweenness_centrality_std = statistics.stdev(betweenness_centrality.values())
            graph_properties[name]['stdevBetweennessCentrality'] = betweenness_centrality_std

            closeness_centrality = nx.closeness_centrality(graph)
            average_closeness_centrality = sum(closeness_centrality.values()) / len(closeness_centrality)
            graph_properties[name]['averageClosenessCentrality'] = average_closeness_centrality
            closeness_centrality_std = statistics.stdev(closeness_centrality.values())
            graph_properties[name]['stdevClosenessCentrality'] = closeness_centrality_std

            try:
                eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000)
                average_eigenvector_centrality = sum(eigenvector_centrality.values()) / len(eigenvector_centrality)
                graph_properties[name]['averageEigenvectorCentrality'] = average_eigenvector_centrality
                eigenvector_centrality_std = statistics.stdev(eigenvector_centrality.values())
                graph_properties[name]['stdevEigenvectorCentrality'] = eigenvector_centrality_std
            except PowerIterationFailedConvergence:
                pass

            assortativity = nx.degree_assortativity_coefficient(graph)
            graph_properties[name]['assortativity'] = assortativity

            density = nx.density(graph)
            graph_properties[name]['density'] = density

            diameter = nx.diameter(graph)
            graph_properties[name]['diameter'] = diameter

            eccentricity = nx.eccentricity(graph)
            average_eccentricity = sum(eccentricity.values()) / len(eccentricity)
            graph_properties[name]['averageEccentricity'] = average_eccentricity
            eccentricity_std = statistics.stdev(eccentricity.values())
            graph_properties[name]['stdevEccentricity'] = eccentricity_std

            pagerank = nx.pagerank(graph)
            average_pagerank = sum(pagerank.values()) / len(pagerank)
            graph_properties[name]['averagePageRank'] = average_pagerank
            pagerank_std = statistics.stdev(pagerank.values())
            graph_properties[name]['stdevPageRank'] = pagerank_std

            hubs, authorities = nx.hits(graph)
            average_hub_score = sum(hubs.values()) / len(hubs)
            graph_properties[name]['averageHubScore'] = average_hub_score
            hub_score_std = statistics.stdev(hubs.values())
            graph_properties[name]['stdevHubScore'] = hub_score_std
            average_authority_score = sum(authorities.values()) / len(authorities)
            graph_properties[name]['averageAuthorityScore'] = average_authority_score
            authority_score_std = statistics.stdev(authorities.values())
            graph_properties[name]['stdevAuthorityScore'] = authority_score_std

            average_neighbor_degree = nx.average_neighbor_degree(graph)
            average_nearest_neighbors_degree = sum(average_neighbor_degree.values()) / len(average_neighbor_degree)
            graph_properties[name]['averageNearestNeighborsDegree'] = average_nearest_neighbors_degree
            nearest_neighbors_degree_std = statistics.stdev(average_neighbor_degree.values())
            graph_properties[name]['stdevNearestNeighborsDegree'] = nearest_neighbors_degree_std

            transitivity = nx.transitivity(graph)
            graph_properties[name]['transitivity'] = transitivity

            node_connectivity = nx.node_connectivity(graph)
            graph_properties[name]['nodeConnectivity'] = node_connectivity

            edge_connectivity = nx.edge_connectivity(graph)
            graph_properties[name]['edgeConnectivity'] = edge_connectivity

            rich_club_coefficient = nx.rich_club_coefficient(graph, normalized=False)
            average_rich_club_coefficient = sum(rich_club_coefficient.values()) / len(rich_club_coefficient)
            graph_properties[name]['averageRichClubCoefficient'] = average_rich_club_coefficient
            rich_club_coefficient_std = statistics.stdev(rich_club_coefficient.values())
            graph_properties[name]['stdevRichClubCoefficient'] = rich_club_coefficient_std

    compute_metrics()

    if selectSubset:
        # Retrieve all the keys from the graph_properties dictionary
        all_keys_set = set()
        for value in graph_properties.values():
            all_keys_set.update(value.keys())
        metric_keys_set = all_keys_set - all_keys_before_set
        metric_keys = list(metric_keys_set)
        metric_keys.sort()

        prompt_text = "Select which metric to approximate.\n"

        # Generate the mapping of numbers to key names
        mapping = "\n".join([f"{i + 1} - {key}" for i, key in enumerate(metric_keys)])

        # Append the mapping to the prompt text
        prompt_text += mapping + "\n"

        choices = [str(num) for num in range(1, len(metric_keys) + 1)]
        # Prompt the user for input
        choice = click.prompt(prompt_text, type=click.Choice(choices))
        chosen_key = metric_keys[int(choice) - 1]
        sorted_graph_properties = dict(sorted(graph_properties.items(), key=lambda x: x[1][chosen_key]))
        sorted_graphs = {name: graphs[name] for name in sorted_graph_properties.keys()}
        sorted_chosen_values = [item[chosen_key] for item in sorted_graph_properties.values()]
        print(f'Values of {chosen_key} for graphs:\n'
              f'Minimum:{min(sorted_chosen_values)}\n'
              f'Maximum:{max(sorted_chosen_values)}')

        value_to_approximate \
            = click.prompt('Enter the value to approximate', type=type(sorted_chosen_values[0]))

        closest_graph_properties_subset = {}
        closest_graphs_subset = {}
        for i in range(subset_size):
            closest_value = min(sorted_chosen_values, key=lambda x: abs(x - value_to_approximate))
            index = sorted_chosen_values.index(closest_value)
            closest_graph_name = list(sorted_graphs.keys())[index]
            closest_graph_properties = sorted_graph_properties[closest_graph_name]
            closest_graph = sorted_graphs[closest_graph_name]

            closest_graph_properties_subset[closest_graph_name] = closest_graph_properties
            closest_graphs_subset[closest_graph_name] = closest_graph
            sorted_chosen_values.pop(index)
            del sorted_graphs[closest_graph_name]
            del sorted_graph_properties[closest_graph_name]

        graphs = closest_graphs_subset
        graph_properties = closest_graph_properties_subset

    choice = click.prompt('Do you want to save the graphs as adjacency lists or create a k8s graph resource?',
                          type=click.Choice(['adj', 'k8s']), default='k8s')
    if choice == 'adj':
        prefix = click.prompt('Enter the file name prefix', type=str, default="")
        for name, graph in graphs.items():
            save_graph_as_adj_list(graph, f'{prefix}_{name}')
    elif choice == 'k8s':
        # generate k8s graph resource .yaml file
        graph_type_string = get_graph_type_long(graph_type)

        # optional value list for node value assignment
        value_list = None
        custom = click.confirm('Do you want to set custom node values?', default=False)
        if custom:
            node_count = max(len(graph.nodes) for name, graph in graphs.items())
            while True:
                value_list_input = click.prompt(f"Enter {node_count} numbers (separated by commas)")
                try:
                    value_list = [int(num.strip()) for num in value_list_input.split(",")]
                except:
                    print('Invalid input. Please try again.')
                if len(value_list) == node_count:
                    break
                else:
                    print('Not the right number of values. Please try again.')

        # optional name graph name setting
        choice = click.prompt(
            'Do you want to: \n' +
            '- use default prefixes [default]\n' +
            '- add a prefix to the default prefix [add]\n' +
            '- specify an own name prefix? [own]',
            type=click.Choice(['default', 'add', 'own']), default="default")
        if choice == 'add':
            added_prefix = click.prompt('Enter the name prefix to add', type=str)
        if choice == 'own':
            own_prefix = click.prompt('Enter the name prefix that will be used as a replacement', type=str)

        series_label = click.prompt('Enter series label (optional)', type=str, default="")
        if not series_label:
            series_label = None

        i = 0
        for name, graph in graphs.items():
            # create name string
            if choice == 'add':
                name_string = added_prefix + name
            elif choice == 'own':
                name_string = f'{own_prefix}_{i + 1}'
            else:
                name_string = name
            # create value list string
            value_list_string = None
            if value_list is not None:
                value_list_string = ','.join(str(value) for value in value_list[:graph.number_of_nodes()])

            # make sure that the name is max. length of 63 for k8s object compatibility
            if len(name_string) > 63:
                name_string = name_string[-63:]

            # adj list string and graph properties
            adjacency_list_string = ','.join(nx.generate_adjlist(graph))
            graph_props = graph_properties[name]

            # generate yaml string
            yaml_string = generate_graph_resource_yaml(name_string,
                                                       adjacency_list_string,
                                                       graph_type_string,
                                                       graph_props,
                                                       series_label,
                                                       value_list_string,
                                                       )
            # save the graph resource as yaml file
            save_graph_resource_yaml(yaml_string, name_string)

            if visualize:
                # export in gephi format
                directory = './generated_gexf/'
                os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
                nx.write_gexf(graph, f"./generated_gexf/{name_string}.gexf")

            i += 1


if __name__ == '__main__':
    generate_graphs()
