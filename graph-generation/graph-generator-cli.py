import io
import os
import click
import networkx as nx
import pygraphviz as pgv
import random
import community
import colorsys
from PIL import Image
from cfg import *
import random_modular_generator_variable_modules as rmg
import sequence_generator as sg


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
    directory = './generated_graphs/'
    os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
    # Save the graph as an adjacency list
    with open(f'./generated_graphs/{name}.adj-list', 'w') as f:
        for line in nx.generate_adjlist(graph):
            f.write(line + ',')


def get_params_simple_graph():
    node_count = click.prompt('Enter the total number of nodes in the graph', type=int)
    comm_count = click.prompt('Enter the number of communities in the graph', type=int)
    equal_sized = click.confirm('Should the communities be of equal size?')
    p_intra = click.prompt('Enter the probability of intra-community edges (0 to 1)', type=float, default=0.4)
    p_inter = click.prompt('Enter the probability of inter-community edges (0 to 1)', type=float, default=0.001)

    return node_count, comm_count, equal_sized, p_intra, p_inter


def get_distribution(distribution_name):
    distributions = {
        POISSON_DISTRIBUTION_NAME: sg.poisson_sequence,
        REGULAR_DISTRIBUTION_NAME: sg.regular_sequence,
        GEOMETRIC_DISTRIBUTION_NAME: sg.geometric_sequence,
        SCALE_FREE_DISTRIBUTION_NAME: sg.scalefree_sequence,
    }
    distribution = distributions.get(distribution_name)
    return distribution


def get_params_complex_graph():
    node_count = click.prompt('Enter the total number of nodes in the graph', type=int)
    degree = click.prompt('Enter the graph degree', type=int)
    comm_count = click.prompt('Enter the number of communities in the graph', type=int)
    modularity = click.prompt('Enter the graph modularity (0 to 1)', type=float, default=0.8)
    choice_degree_distribution = click.prompt('Enter the degree distribution function',
                                              type=click.Choice([REGULAR_DISTRIBUTION_NAME, POISSON_DISTRIBUTION_NAME,
                                                                 GEOMETRIC_DISTRIBUTION_NAME,
                                                                 SCALE_FREE_DISTRIBUTION_NAME]),
                                              default=POISSON_DISTRIBUTION_NAME)
    community_degree_distribution = click.prompt('Enter the community distribution function',
                                                 type=click.Choice(
                                                     [REGULAR_DISTRIBUTION_NAME, POISSON_DISTRIBUTION_NAME,
                                                      GEOMETRIC_DISTRIBUTION_NAME, SCALE_FREE_DISTRIBUTION_NAME]),
                                                 default=REGULAR_DISTRIBUTION_NAME)

    return node_count, degree, comm_count, modularity, \
           get_distribution(choice_degree_distribution), \
           get_distribution(community_degree_distribution)


def get_params_scale_free_graph():
    node_count = click.prompt('Enter the total number of nodes in the graph', type=int)
    print('Now alpha, betta and gamma can be defined, their sum must be 1')
    while True:
        alpha = click.prompt('Enter alpha (0-1)', type=float, default=0.9)
        beta = click.prompt('Enter beta (0-1)', type=float, default=0.05)
        gamma = click.prompt('Enter gamma (0-1)', type=float, default=0.05)
        if alpha + beta + gamma == 1:
            break
        else:
            print('Sum is not 1, try again')
    return node_count, alpha, beta, gamma


def get_params_barabasi_albert_graph():
    node_count = click.prompt('Enter the total number of nodes in the graph', type=int)
    edge_degree = click.prompt('Enter the edge degree in the graph', type=int, default=1)
    return node_count, edge_degree


def get_get_params_func(graph_type):
    get_params_funcs = {
        GRAPH_TYPE_SIMPLE: get_params_simple_graph,
        GRAPH_TYPE_COMPLEX: get_params_complex_graph,
        GRAPH_TYPE_SCALE_FREE: get_params_scale_free_graph,
        GRAPH_TYPE_BARABASI_ALBERT: get_params_barabasi_albert_graph,
    }
    func = get_params_funcs.get(graph_type)
    return func


def get_simple_graph_name(node_count, comm_count, equal_sized, p_intra, p_inter):
    eq_str = 'eq' if equal_sized else 'ne'
    return f'SIMPL_n{node_count}_c{comm_count}_{eq_str}_p1_{p_intra}_p2_{p_inter}'


def generate_simple_graph(node_count, comm_count, equal_sized, p_intra, p_inter):
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
        nodes = range(sum(sizes[:i]), sum(sizes[:i + 1]))
        # changed to connected watts strogatz graphs from random qnp
        # because they are always interconnected
        # subgraph = nx.gnp_random_graph(sizes[i], p=p_intra)
        subgraph = nx.connected_watts_strogatz_graph(sizes[i], k=int(sizes[i] * p_intra), p=1)
        mapping = dict(zip(range(sizes[i]), nodes))
        subgraph = nx.relabel_nodes(subgraph, mapping)
        graph.add_edges_from(subgraph.edges())

    community_dict = {}
    for i in range(comm_count):
        nodes = range(sum(sizes[:i]), sum(sizes[:i + 1]))
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

    return graph


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


# https://github.com/bansallab/modular_graph_generator/blob/master/mock_code.py
def get_short_distribution_string(distribution):
    return 'R'


def get_complex_graph_name(node_count, degree, comm_count, modularity, degree_distribution, module_distribution):
    dd = get_short_distribution_string(degree_distribution)
    md = get_short_distribution_string(module_distribution)
    return f'COMPL_n{node_count}_d{degree}_c{comm_count}_m_{modularity}_dd{dd}_md{md}'


def generate_complex_graph(node_count, degree, comm_count, modularity, degree_distribution, module_distribution):
    degree_function = sg.poisson_sequence
    module_function = sg.regular_sequence
    "Generating graph....."
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
        GRAPH_TYPE_SIMPLE: generate_simple_graph,
        GRAPH_TYPE_COMPLEX: generate_complex_graph,
        GRAPH_TYPE_SCALE_FREE: generate_scale_free_graph,
        GRAPH_TYPE_BARABASI_ALBERT: generate_barabasi_albert_graph,
    }
    func = create_graph_funcs.get(graph_type)
    return func


def get_graph_name(graph_type, graph_params):
    get_graph_name_funcs = {
        GRAPH_TYPE_SIMPLE: get_simple_graph_name,
        GRAPH_TYPE_COMPLEX: get_complex_graph_name,
        GRAPH_TYPE_SCALE_FREE: get_scale_free_graph_name,
        GRAPH_TYPE_BARABASI_ALBERT: get_barabasi_albert_graph_name,
    }
    func = get_graph_name_funcs.get(graph_type)
    return func(*graph_params)


@click.command()
@click.option('--graph-type',
              type=click.Choice([f'{GRAPH_TYPE_SIMPLE}', f'{GRAPH_TYPE_COMPLEX}',
                                 f'{GRAPH_TYPE_SCALE_FREE}', f'{GRAPH_TYPE_BARABASI_ALBERT}']),
              help=f'The graph type (simple, complex, scale-free or barabasi-albert) for the created graph',
              prompt='Choose graph type:\n' +
                     f'* [{GRAPH_TYPE_SIMPLE}] : Simple modular graph creation based on inter/intra-edge generation\n' +
                     f'* [{GRAPH_TYPE_COMPLEX}] : Complex modular graph creation based on target modularity\n' +
                     f'* [{GRAPH_TYPE_SCALE_FREE}] : Scale-free graph creation\n' +
                     f'* [{GRAPH_TYPE_BARABASI_ALBERT}] : Scale-free graph creation\n')
@click.option('--count',
              type=int,
              default=1,
              help='The number of graphs generated',
              prompt='Choose the number of graphs that are to be generated')
def generate_graphs(graph_type, count):
    get_params_func = get_get_params_func(graph_type)
    create_graph_func = get_creation_func(graph_type)
    graph_params = get_params_func()
    graph = create_graph_func(*graph_params)
    graphs = [(get_graph_name(graph_type, graph_params), graph)]

    if count > 1:
        same_params = click.confirm('Do you want to initialize all graphs with the same parameters?')

        if same_params:
            for _ in range(0, count - 1):
                graph = create_graph_func(*graph_params)
                graphs.append((get_graph_name(graph_type, graph_params), graph))

        else:
            one_by_one = click.confirm('Do you want to initialize each following graph one by one?')

            if one_by_one:
                for _ in range(0, count - 1):
                    graph_params = get_params_func()
                    graph = create_graph_func(*graph_params)
                    graphs.append((get_graph_name(graph_type, graph_params), graph))
            else:
                print('todo')
                # start / end values for params

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
        prefix = click.prompt('Enter the file name prefix', type=str)
        for name, graph in graphs:
            save_graph_as_adj_list(graph, f'{prefix}_{name}')
    if choice == 'k8s':
        print('todo')
        # generate k8s graph resource .yaml file


if __name__ == '__main__':
    generate_graphs()
