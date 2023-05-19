import io
import os
import click
import networkx as nx
import numpy as np
import pygraphviz as pgv
import random
import community
import colorsys
from PIL import Image
from cfg import *
import random_modular_generator_variable_modules as rmg
import sequence_generator as sg
import yaml


def make_fully_interconnected(graph):
    components = list(nx.connected_components(graph))
    num_components = len(components)

    if num_components < 2:
        return graph  # No need to add edges if the graph is already fully interconnected

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

    return make_fully_interconnected(new_graph)


def apply_louvain(graph):
    """
       Apply Louvain algorithm for community detection on the given graph.
       Sets the attribute 'louvain_community'.

       Args:
           graph (NetworkX graph): The graph to apply the algorithm to.

       Returns:
           None
       """
    # apply the Louvain method to detect communities
    partition = community.best_partition(graph)

    # mark the communities in the graph
    for node, community_id in partition.items():
        graph.nodes[node]['louvain_community'] = community_id

    return max(partition.values()) + 1


def compute_modularity(graph, use_louvain=False):
    """
      Compute the modularity of the given graph.

      Args:
          graph (NetworkX graph): The graph to compute the modularity of.
          use_louvain (bool, optional): Whether to use Louvain algorithm for community detection. Default is False.

      Returns:
          float: The modularity of the graph.
    """

    column_name = 'louvain_community' if use_louvain else 'community'
    communities = [set(node for node, attr in graph.nodes(data=True) if attr[column_name] == c)
                   for c in set(nx.get_node_attributes(graph, column_name).values())]

    # Compute the modularity
    return nx.algorithms.community.modularity(graph, communities)


def generate_colors(num_colors):
    """
    Generate an array of distinct colors based on an input size.

    Args:
        num_colors (int): The number of colors to generate.

    Returns:
        colors (list of tuples): A list of RGB tuples, each representing a distinct color.
    """
    colors = []
    for i in range(num_colors):
        # Generate a color with a different hue value for each iteration
        hue = i / float(num_colors)
        # Set the saturation and brightness to fixed values to ensure a consistent look
        saturation = 0.5
        brightness = 0.95
        # Convert the HSV color to an RGB color and append it to the list
        rgb = tuple(int(255 * x) for x in colorsys.hsv_to_rgb(hue, saturation, brightness))
        colors.append(rgb)
    return colors


def plot(graph, num_comm=0, use_louvain=False):
    """
    Plot the given graph.

    Args:
        graph (NetworkX graph): The graph to plot.
        num_comm (int, optional): The number of communities in the graph. Default is 0.
        use_louvain (bool, optional): Whether to use Louvain algorithm for community detection. Default is False.

    Returns:
        None
    """
    # Draw the graph using Graphviz
    g = pgv.AGraph(directed=False)

    if num_comm > 0:
        colors = generate_colors(num_comm)

        # Assign a color to each node based on its community
        community_colors = {}
        for node, data in graph.nodes(data=True):
            community_id = data['louvain_community'] if use_louvain else data['community']
            if community_id not in community_colors:
                community_colors[community_id] = colors.pop(0)
            node_color = '#' + ''.join(format(c, '02x') for c in community_colors[community_id])
            g.add_node(node, style='filled', fillcolor=node_color)

    # Add edges to the graph
    for edge in graph.edges():
        g.add_edge(edge[0], edge[1])

    # Layout the graph
    g.layout(prog='dot')

    # Draw the graph to a byte buffer
    buffer = io.BytesIO()
    g.draw(buffer, format='png')

    # Open the image using PIL
    image = Image.open(buffer)

    # Display the image
    image.show()


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
    return click.prompt('Enter the total number of nodes in the graph', type=int)


def prompt_comm_count():
    return click.prompt('Enter the number of communities in the graph', type=int)


def prompt_p_intra():
    return click.prompt('Enter the probability of intra-community edges (0 to 1)', type=float, default=0.4)


def prompt_p_inter():
    return click.prompt('Enter the probability of inter-community edges (0 to 1)', type=float, default=0.001)


def prompt_simple_equal_comms():
    return click.confirm('Should the communities be of equal size?')


def get_params_simple_graph():
    node_count = prompt_node_count()
    comm_count = prompt_comm_count()
    equal_sized = prompt_simple_equal_comms()
    p_intra = prompt_p_intra()
    p_inter = prompt_p_inter()
    return node_count, comm_count, p_intra, p_inter, equal_sized


def get_graph_properties_simple_graph(node_count, comm_count, p_intra, p_inter, equal_sized):
    return {
        'nodeCount': node_count,
        'communityCount': comm_count,
        'probabilityIntraCommunityEdge': p_intra,
        'probabilityInterCommunityEdge': p_inter,
        'equalSizedCommunities': equal_sized
    }


def get_end_params_simple_graph():
    node_count = prompt_node_count()
    comm_count = prompt_comm_count()
    p_intra = prompt_p_intra()
    p_inter = prompt_p_inter()
    return node_count, comm_count, p_intra, p_inter


def get_distribution(distribution_name):
    distributions = {
        POISSON_DISTRIBUTION_NAME: sg.poisson_sequence,
        REGULAR_DISTRIBUTION_NAME: sg.regular_sequence,
        GEOMETRIC_DISTRIBUTION_NAME: sg.geometric_sequence,
        SCALE_FREE_DISTRIBUTION_NAME: sg.scalefree_sequence,
    }
    distribution = distributions.get(distribution_name)
    return distribution


def prompt_degree():
    return click.prompt('Enter the graph degree', type=int)


def prompt_modularity():
    return click.prompt('Enter the graph modularity (0 to 1)', type=float, default=0.8)


def prompt_degree_distribution():
    return click.prompt('Enter the degree distribution function',
                        type=click.Choice([REGULAR_DISTRIBUTION_NAME, POISSON_DISTRIBUTION_NAME,
                                           GEOMETRIC_DISTRIBUTION_NAME,
                                           SCALE_FREE_DISTRIBUTION_NAME]),
                        default=POISSON_DISTRIBUTION_NAME)


def prompt_community_distribution():
    return click.prompt('Enter the community distribution function',
                        type=click.Choice(
                            [REGULAR_DISTRIBUTION_NAME, POISSON_DISTRIBUTION_NAME,
                             GEOMETRIC_DISTRIBUTION_NAME, SCALE_FREE_DISTRIBUTION_NAME]),
                        default=REGULAR_DISTRIBUTION_NAME)


def get_params_complex_graph():
    node_count = prompt_node_count()
    degree = prompt_degree()
    comm_count = prompt_comm_count()
    modularity = prompt_modularity()
    degree_distribution = prompt_degree_distribution()
    community_distribution = prompt_community_distribution()

    return node_count, degree, comm_count, modularity, get_distribution(degree_distribution), get_distribution(
        community_distribution)


def get_graph_properties_complex_graph(node_count, degree, comm_count, modularity, degree_distribution,
                                       community_distribution):
    return {
        'nodeCount': node_count,
        'degree': degree,
        'communityCount': comm_count,
        'attemptedModularity': modularity,
        'degreeDistributionFunction': get_long_distribution_string(degree_distribution),
        'communityDistributionFunction': get_long_distribution_string(community_distribution)
    }


def get_end_params_complex_graph():
    node_count = prompt_node_count()
    degree = prompt_degree()
    comm_count = prompt_comm_count()
    modularity = prompt_modularity()

    return node_count, degree, comm_count, modularity


def prompt_alpha():
    return click.prompt('Enter alpha (0-1)', type=float, default=0.9)


def prompt_beta():
    return click.prompt('Enter beta (0-1)', type=float, default=0.05)


def prompt_gamma():
    return click.prompt('Enter gamma (0-1)', type=float, default=0.05)


def get_params_scale_free_graph():
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
    return node_count, alpha, beta, gamma


def get_graph_properties_scale_free_graph(node_count, alpha, beta, gamma):
    return {
        'nodeCount': node_count,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma
    }


def prompt_edge_degree():
    return click.prompt('Enter the edge degree in the graph', type=int, default=1)


def get_params_barabasi_albert_graph():
    node_count = prompt_node_count()
    edge_degree = prompt_edge_degree()
    return node_count, edge_degree


def get_graph_properties_barabasi_albert_graph(node_count, edge_degree):
    return {
        'nodeCount': node_count,
        'edgeDegree': edge_degree
    }


def get_get_params_func(graph_type):
    get_params_funcs = {
        GRAPH_TYPE_SIMPLE_SHORT: get_params_simple_graph,
        GRAPH_TYPE_COMPLEX_SHORT: get_params_complex_graph,
        GRAPH_TYPE_SCALE_FREE_SHORT: get_params_scale_free_graph,
        GRAPH_TYPE_BARABASI_ALBERT_SHORT: get_params_barabasi_albert_graph
    }
    func = get_params_funcs.get(graph_type)
    return func


def get_get_graph_properties_func(graph_type):
    get_graph_properties_funcs = {
        GRAPH_TYPE_SIMPLE_SHORT: get_graph_properties_simple_graph,
        GRAPH_TYPE_COMPLEX_SHORT: get_graph_properties_complex_graph,
        GRAPH_TYPE_SCALE_FREE_SHORT: get_graph_properties_scale_free_graph,
        GRAPH_TYPE_BARABASI_ALBERT_SHORT: get_graph_properties_barabasi_albert_graph
    }
    func = get_graph_properties_funcs.get(graph_type)
    return func


def get_get_end_params_func(graph_type):
    get_end_params_funcs = {
        GRAPH_TYPE_SIMPLE_SHORT: get_end_params_simple_graph,
        GRAPH_TYPE_COMPLEX_SHORT: get_end_params_complex_graph,
        GRAPH_TYPE_SCALE_FREE_SHORT: get_params_scale_free_graph,
        GRAPH_TYPE_BARABASI_ALBERT_SHORT: get_params_barabasi_albert_graph
    }
    func = get_end_params_funcs.get(graph_type)
    return func


def get_simple_graph_name(node_count, comm_count, equal_sized, p_intra, p_inter):
    eq_str = 'eq' if equal_sized else 'ne'
    return f'SIMPL_n{node_count}_c{comm_count}_{eq_str}_p1_{p_intra}_p2_{p_inter}'


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
    # this does not work (TODO: FIX)
    # reproduce: node_count: 3, communities: 2, not equalized
    for i in range(comm_count):
        nodes = list(range(sum(sizes[:i]), sum(sizes[:i+1])))
        size = sizes[i]
        subgraph = nx.gnp_random_graph(size, p=p_intra)
        mapping = dict(zip(range(size), nodes))
        subgraph = nx.relabel_nodes(subgraph, mapping)
        graph.add_nodes_from(subgraph.nodes())
        graph.add_edges_from(subgraph.edges())

    community_dict = {}
    for i in range(comm_count):
        nodes = list(range(sum(sizes[:i]), sum(sizes[:i+1])))
        for node in nodes:
            community_dict[node] = i

    # Set the community attribute for each node in the graph
    nx.set_node_attributes(graph, community_dict, 'community')

    # guarantee that the graph is connected
    for i in range(comm_count - 1):
        add_random_inter_community_edge(graph, i, i + 1)

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
    nodes_i = [n for n in graph.nodes if graph.nodes[n]['community'] == i]
    nodes_j = [n for n in graph.nodes if graph.nodes[n]['community'] == j]
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
    nodes_i = [n for n in graph.nodes if graph.nodes[n]['community'] == i]
    nodes_j = [n for n in graph.nodes if graph.nodes[n]['community'] == j]
    node_i = random.choice(nodes_i)
    node_j = random.choice(nodes_j)
    graph.add_edge(node_i, node_j)
    return graph


def get_scale_free_graph_name(node_count, alpha, beta, gamma):
    return f'SCALE_n{node_count}_a{alpha}_b{beta}_g{gamma}'


def generate_scale_free_graph(node_count, alpha, beta, gamma):
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
          It affects the level of connectivity in the resulting graph

      Returns:
          A fully-interconnected scale-free graph of `node_count` nodes,
          generated using the networkx scale_free_graph implementation
          with the specified `alpha`, `beta`, and `gamma`.
          Self-loops are also removed from the resulting graph.
      """
    # Generate a scale-free graph with the specified number of nodes and parameters.
    graph = nx.scale_free_graph(node_count, alpha=alpha, beta=beta, gamma=gamma)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    graph_undirected = nx.to_undirected(graph)
    fully_interconnected_graph = make_fully_interconnected(graph_undirected)
    return fully_interconnected_graph


def get_short_distribution_string(distribution):
    short_descriptions = {
        sg.poisson_sequence: 'P',
        sg.regular_sequence: 'R',
        sg.geometric_sequence: 'G',
        sg.scalefree_sequence: 'S',
    }
    description = short_descriptions.get(distribution)
    return description


def get_long_distribution_string(distribution):
    long_descriptions = {
        sg.poisson_sequence: 'PoissonSequence',
        sg.regular_sequence: 'RegularSequence',
        sg.geometric_sequence: 'GeometricSequence',
        sg.scalefree_sequence: 'ScaleFreeSequence',
    }
    description = long_descriptions.get(distribution)
    return description


def get_complex_graph_name(node_count, degree, comm_count, modularity, degree_distribution, module_distribution):
    dd = get_short_distribution_string(degree_distribution)
    md = get_short_distribution_string(module_distribution)
    return f'COMPL_n{node_count}_d{degree}_c{comm_count}_m_{modularity}_dd{dd}_md{md}'


def generate_complex_graph(node_count, degree, comm_count, modularity, degree_distribution, module_distribution):
    degree_function = sg.poisson_sequence
    module_function = sg.regular_sequence
    graph = rmg.generate_modular_networks(node_count, degree_function, module_function, modularity, comm_count, degree)
    return graph


def get_barabasi_albert_graph_name(node_count, edge_degree):
    return f'SCALE_n{node_count}_e{edge_degree}'


def generate_barabasi_albert_graph(node_count, edge_degree):
    """
       Generates a Barabasi-Albert graph with the specified number of nodes and minimum degree.

       Args:
           node_count (int): The number of nodes in the graph.
           edge_degree (int): The number of edges to attach from a new node to existing nodes.

       Returns:
           A Barabasi-Albert graph of `node_count` nodes and `edge_degree` node attachments.
   """
    # Generate a barabasi albert graph with the specified number of nodes and exponent.
    graph = nx.barabasi_albert_graph(node_count, edge_degree)
    return graph


def get_creation_func(graph_type):
    create_graph_funcs = {
        GRAPH_TYPE_SIMPLE_SHORT: generate_simple_graph,
        GRAPH_TYPE_COMPLEX_SHORT: generate_complex_graph,
        GRAPH_TYPE_SCALE_FREE_SHORT: generate_scale_free_graph,
        GRAPH_TYPE_BARABASI_ALBERT_SHORT: generate_barabasi_albert_graph,
    }
    func = create_graph_funcs.get(graph_type)
    return func


def get_graph_name(graph_type, graph_params):
    get_graph_name_funcs = {
        GRAPH_TYPE_SIMPLE_SHORT: get_simple_graph_name,
        GRAPH_TYPE_COMPLEX_SHORT: get_complex_graph_name,
        GRAPH_TYPE_SCALE_FREE_SHORT: get_scale_free_graph_name,
        GRAPH_TYPE_BARABASI_ALBERT_SHORT: get_barabasi_albert_graph_name,
    }
    func = get_graph_name_funcs.get(graph_type)
    return func(*graph_params)


def generate_graph_resource_yaml(name, adjacency_list, graph_type, graph_properties, value_list=None):
    resource_dict = {
        'apiVersion': 'gossip.io/v1',
        'kind': 'Graph',
        'metadata': {
            'name': name
        },
        'spec': {
            'adjacencyList': adjacency_list,
            'graphType': graph_type,
            'graphProperties': graph_properties
        }
    }

    if value_list is not None:
        resource_dict['spec']['valueList'] = value_list

    return yaml.dump(resource_dict)


def save_graph_resource_yaml(content, name):
    directory = './generated_yaml/'
    os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
    # Save the graph as an adjacency list
    with open(f'./generated_yaml/{name}.yaml', 'w') as f:
        f.write(content)


def get_graph_type_long(graph_type):
    long_graph_types = {
        GRAPH_TYPE_SIMPLE_SHORT: GRAPH_TYPE_SIMPLE,
        GRAPH_TYPE_COMPLEX_SHORT: GRAPH_TYPE_COMPLEX,
        GRAPH_TYPE_SCALE_FREE_SHORT: GRAPH_TYPE_SCALE_FREE,
        GRAPH_TYPE_BARABASI_ALBERT_SHORT: GRAPH_TYPE_BARABASI_ALBERT,
    }
    return long_graph_types.get(graph_type)


@click.command()
@click.option('--graph-type',
              type=click.Choice([f'{GRAPH_TYPE_SIMPLE_SHORT}', f'{GRAPH_TYPE_COMPLEX_SHORT}',
                                 f'{GRAPH_TYPE_SCALE_FREE_SHORT}', f'{GRAPH_TYPE_BARABASI_ALBERT_SHORT}']),
              help=f'The graph type (simple, complex, scale-free or barabasi-albert) for the created graph',
              prompt='Choose graph type:\n' +
                     f'* [{GRAPH_TYPE_SIMPLE_SHORT}] : Simple modular graph creation based on inter/intra-edge generation\n' +
                     f'* [{GRAPH_TYPE_COMPLEX_SHORT}] : Complex modular graph creation based on target modularity\n' +
                     f'* [{GRAPH_TYPE_SCALE_FREE_SHORT}] : Scale-free graph creation\n' +
                     f'* [{GRAPH_TYPE_BARABASI_ALBERT_SHORT}] : Scale-free graph creation\n')
@click.option('--count',
              type=int,
              default=1,
              help='The number of graphs generated',
              prompt='Choose the number of graphs that are to be generated')
def generate_graphs(graph_type, count):
    get_params_func = get_get_params_func(graph_type)
    get_graph_properties_func = get_get_graph_properties_func(graph_type)
    get_end_params_func = get_get_end_params_func(graph_type)
    create_graph_func = get_creation_func(graph_type)
    graph_params = get_params_func()
    graph = create_graph_func(*graph_params)
    graph_properties = [get_graph_properties_func(*graph_params)]
    graphs = [(get_graph_name(graph_type, graph_params), graph)]

    if count > 1:
        same_params = click.confirm('Do you want to initialize all graphs with the same parameters?')

        if same_params:
            for _ in range(0, count - 1):
                graph = create_graph_func(*graph_params)
                graph_properties.append(get_graph_properties_func(*graph_params))
                graphs.append((get_graph_name(graph_type, graph_params), graph))

        else:
            one_by_one = click.confirm('Do you want to initialize each following graph one by one?')

            if one_by_one:
                for _ in range(0, count - 1):
                    graph_params = get_params_func()
                    graph = create_graph_func(*graph_params)
                    graph_properties.append(get_graph_properties_func(*graph_params))
                    graphs.append((get_graph_name(graph_type, graph_params), graph))
            else:
                end_params = get_end_params_func()

                param_lists = []
                for i in range(0, len(graph_params)):
                    if i < len(end_params):
                        start = graph_params[i]
                        end = end_params[i]
                        values = np.linspace(start, end, count)[1:]
                    else:
                        values = np.full(count - 1, graph_params[i])
                    param_lists.append(values)

                # Transpose the input array to align the i-th elements from each inner array
                transposed_params = np.transpose(param_lists)
                # Create a new structure with tuples of i-th elements from each inner array
                graph_params_list = [tuple(row) for row in transposed_params]

                for params in graph_params_list:
                    graph = create_graph_func(*params)
                    graph_properties.append(get_graph_properties_func(*params))
                    graphs.append((get_graph_name(graph_type, params), graph))

    # all graphs are created
    # rename duplicate graph names
    renamed_graphs = []
    name_counts = {}

    for name, graph in graphs:
        if name not in name_counts:
            # First occurrence of the name
            name_counts[name] = 1
            renamed_graphs.append((name, graph))
        else:
            # Duplicate name found
            count = name_counts[name]
            new_name = f"{name}_{count + 1}"
            renamed_graphs.append((new_name, graph))
            name_counts[name] += 1
    graphs = renamed_graphs

    visualize = click.confirm('Do you want to see the created graphs?')
    if visualize:
        for name, graph in graphs:
            num_communities = apply_louvain(graph)
            computed_modularity = compute_modularity(graph, use_louvain=True)
            print(f'The graph has a computed modularity of {computed_modularity}.')
            plot(graph, num_communities, use_louvain=True)

    choice = click.prompt('Do you want to save the graphs as adjacency lists or create a k8s graph resource?',
                          type=click.Choice(['adj', 'k8s']))
    if choice == 'adj':
        prefix = click.prompt('Enter the file name prefix', type=str, default="")
        for name, graph in graphs:
            save_graph_as_adj_list(graph, f'{prefix}_{name}')
    elif choice == 'k8s':
        # generate k8s graph resource .yaml file
        graph_type_string = get_graph_type_long(graph_type)

        value_list_string = None
        choice = click.prompt(
            'Do you want to assign random node values, use the node number as its value or ' +
            'assign custom values yourself?',
            type=click.Choice(['rand', 'own', 'custom']))
        node_count = graphs[0].number_of_nodes()
        if choice == 'own':
            value_list_string = ','.join(str(i) for i in range(1, node_count + 1))
        if choice == 'custom':
            while True:
                value_list_string = click.prompt(f"Enter {node_count} numbers (separated by commas)")
                numbers = [int(num.strip()) for num in value_list_string.split(",")]
                if len(numbers == node_count):
                    break
                else:
                    print('Not the right number of values. Please try again.')

        choice = click.prompt('Do you want to use default prefixes, add a prefix to the default prefix or ' +
                              'specify an own name prefix?',
                              type=click.Choice(['default', 'add', 'own']))
        if choice == 'add':
            added_prefix = click.prompt('Enter the name prefix to add', type=str)
        if choice == 'own':
            own_prefix = click.prompt('Enter the name prefix that will be used as a replacement', type=str)

        i = 0
        for name, graph in graphs:
            if choice == 'add':
                name_string = added_prefix + name
            elif choice == 'own':
                name_string = own_prefix
            else:
                name_string = name

            # make sure that the name is max. length of 63 for k8s object compatibility
            if len(name_string) > 63:
                name_string = name_string[-63:]

            adjacency_list_string = ','.join(nx.generate_adjlist(graph))
            graph_props = graph_properties[i]
            yaml_string = generate_graph_resource_yaml(name_string,
                                                       adjacency_list_string,
                                                       graph_type_string,
                                                       graph_props,
                                                       value_list_string)
            save_graph_resource_yaml(name, yaml_string)
            i += 1


if __name__ == '__main__':
    generate_graphs()
