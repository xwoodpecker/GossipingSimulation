import os
import random
import statistics
import warnings
from collections import Counter

import click
import community
import networkx as nx
import numpy as np
import powerlaw
import yaml
from networkx import PowerIterationFailedConvergence

# for powerlaw package
np.seterr(divide='ignore', invalid='ignore')
# for nx scipy interaction
warnings.filterwarnings("ignore", category=FutureWarning)


def prompt_node_count():
    """
    Prompt the user to enter the total number of nodes in the graph.

    Returns:
        int: The total number of nodes entered by the user.
    """
    return click.prompt('Enter the total number of nodes in the graph', type=int, default=1000)


def prompt_subgraph_count():
    """
    Prompt the user to enter the total number of subgraphs to be generated.

    Returns:
        int: The total number of subgraphs entered by the user.
    """
    return click.prompt('Enter the total number of subgraphs to be generated', type=int, default=1)


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


def compute_average_node_degree(graph):
    """
    Compute the average edge degree of a graph.

    Parameters:
        graph (networkx.Graph): The input graph.

    Returns:
        float: The average edge degree.

    """
    degrees = dict(graph.degree())
    total_nodes = graph.number_of_nodes()

    average_node_degree = sum(degrees.values()) / total_nodes

    return average_node_degree


def compute_stdev_node_degree(graph):
    """
    Compute the standard deviation edge degree of a graph.

    Parameters:
        graph (networkx.Graph): The input graph.

    Returns:
        float: The standard deviation edge degree.

    """
    degrees = dict(graph.degree())

    stdev_node_degree = statistics.stdev(degrees)

    return stdev_node_degree


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


def extract_random_subgraphs(graph, subgraph_size, num_subgraphs):
    subgraphs = {}

    i = 0
    while len(subgraphs) < num_subgraphs:
        subset = set()
        remaining_nodes = set(graph.nodes())
        start_node = random.choice(list(remaining_nodes))
        subset.add(start_node)
        remaining_nodes.remove(start_node)

        while len(subset) < subgraph_size:
            neighbors = set(graph.neighbors(start_node)).intersection(remaining_nodes)
            if len(neighbors) == 0:
                start_node = random.choice(list(subset))
                continue
            next_node = random.choice(list(neighbors))
            subset.add(next_node)
            remaining_nodes.remove(next_node)
            start_node = next_node

        if len(subset) == subgraph_size:
            subgraph = nx.Graph()
            # Add the subset of nodes to the subgraph
            subgraph.add_nodes_from(subset)
            # Add edges to the subgraph based on the original graph
            for node in subgraph.nodes:
                neighbors = list(graph.neighbors(node))
                subgraph.add_edges_from((node, neighbor) for neighbor in neighbors if neighbor in subgraph)

            name = f'gnutella-n{num_nodes}-{i}'
            subgraphs[name] = subgraph
            print(f"Subgraph {name} created")
            i += 1

    return subgraphs


# Load the Gnutella P2P network from an MTX file
graph = nx.read_edgelist("./gnutella/tech-p2p-gnutella.txt")
graph = nx.convert_node_labels_to_integers(graph)

# Define the number of nodes in each subset and the number of subgraphs
num_nodes = prompt_node_count()
num_subgraphs = prompt_subgraph_count()

# Create the interconnected graphs
# subgraphs = create_interconnected_graphs(graph, num_nodes, num_subgraphs)
graphs = extract_random_subgraphs(graph, num_nodes, num_subgraphs)

# visualize in case the user wants to see the graphs
# visualize = click.confirm('Do you want to see the created graphs?', default=False)
# export to gephi at the end in case the user wants to see the graphs
visualize = click.confirm('Do you want to export the created graphs for visualisation?', default=False)

graph_properties = {}


def compute_metrics():
    print(f'Computing graph metrics...')
    for name, graph in graphs.items():
        print(f'Computing metrics for graph {name}...')
        graph_properties[name] = {}
        graph_properties[name]['nodeCount'] = num_nodes

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

        computed_avg_degree = compute_average_node_degree(graph)
        # print(f'The graph has a computed average degree of {computed_avg_degree}.')
        graph_properties[name]['averagenodeDegree'] = computed_avg_degree
        computed_stdev_degree = compute_stdev_node_degree(graph)
        graph_properties[name]['stdevnodeDegree'] = computed_stdev_degree

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
        average_neighbors_degree = sum(average_neighbor_degree.values()) / len(average_neighbor_degree)
        graph_properties[name]['averageNeighborsDegree'] = average_neighbors_degree
        nearest_neighbors_degree_std = statistics.stdev(average_neighbor_degree.values())
        graph_properties[name]['stdevNeighborsDegree'] = nearest_neighbors_degree_std

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

choice = click.prompt('Do you want to save the graphs as adjacency lists or create a k8s graph resource?',
                      type=click.Choice(['adj', 'k8s']), default='k8s')
if choice == 'adj':
    prefix = click.prompt('Enter the file name prefix', type=str, default="")
    for name, graph in graphs.items():
        save_graph_as_adj_list(graph, f'{prefix}_{name}')
elif choice == 'k8s':
    # generate k8s graph resource .yaml file
    graph_type_string = "real-network"

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

        # generate yaml string
        yaml_string = generate_graph_resource_yaml(name_string,
                                                   adjacency_list_string,
                                                   graph_type_string,
                                                   graph_properties[name],
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
# You now have a list of interconnected graphs, each containing 1000 nodes from the Gnutella P2P network
