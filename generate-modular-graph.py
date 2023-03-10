import networkx as nx
import random
import pygraphviz as pgv
import community
import colorsys


def generate_modular_graph(node_count, comm_count, equal_sized, p_intra, p_inter):
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
        subgraph = nx.gnp_random_graph(sizes[i], p=p_intra)
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
          A scale-free graph of `node_count` nodes, generated using the networkx scale_free_graph implementation
           with the specified `alpha`, `beta`, and `gamma`.
      """
    # Generate a scale-free graph with the specified number of nodes and parameters.
    graph = nx.scale_free_graph(node_count, alpha=alpha, beta=beta, gamma=gamma)
    graph_undirected = nx.to_undirected(graph)

    return graph_undirected


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


def save_graph_as_adj_list(graph, name):
    """
      Save the given graph as an adjacency list.

      Args:
          graph (NetworkX graph): The graph to save.
          name (str): The name of the file to save the adjacency list as.

      Returns:
          None
      """
    # Save the graph as an adjacency list
    with open('./generated_graphs/{}.adj-list'.format(name), 'w') as f:
        for line in nx.generate_adjlist(graph):
            f.write(line + ',')

def save_graph_as_adj_matrix(graph, name):
    """
    Save the given graph as an adjacency matrix.

    Args:
        graph (NetworkX graph): The graph to save.
        name (str): The name of the file to save the adjacency matrix as.

    Returns:
        None
    """
    # Get the adjacency matrix of the graph
    adj_matrix = nx.adjacency_matrix(graph)

    # Write the adjacency matrix to a file
    with open('./generated_graphs/{}.adj-matrix'.format(name), 'w') as f:
        for row in adj_matrix.todense():
            # Convert the row to a comma-separated string of integers
            row_str = ','.join(str(x) for x in row.tolist()[0])
            f.write(row_str + '\n')


def plot(graph, name, num_comm=0, use_louvain=False):
    """
    Plot the given graph.

    Args:
        graph (NetworkX graph): The graph to plot.
        name (str): The name of the plot.
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

    # Draw the graph
    g.draw('./generated_graphs/{}.png'.format(name))


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


# Call the function to create and save the graph

# Create a simple graph
G = nx.Graph()
# Add nodes and edges to the graph
G.add_nodes_from([1, 2, 3, 4])
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])
simple_graph_name = 'simple_graph'
save_graph_as_adj_list(G, simple_graph_name)
save_graph_as_adj_matrix(G, simple_graph_name)
plot(G, simple_graph_name)

# Create a modular graph
modular_graph1 = generate_modular_graph(100, 10, True, 0.4, 0.001)
modular_graph1_name = 'modular_graph1'
apply_louvain(modular_graph1)
print('{} has modularity of {}'.format(modular_graph1_name, compute_modularity(modular_graph1)))
save_graph_as_adj_list(modular_graph1, modular_graph1_name)
plot(modular_graph1, '{}_actual_comm'.format(modular_graph1_name), 10, use_louvain=False)
plot(modular_graph1, '{}_louvain_comm'.format(modular_graph1_name), 10, use_louvain=True)

# Create a scale free graph
alpha = 0.9
beta = 0.95 - alpha
gamma = 0.05
scale_free_graph = generate_scale_free_graph(100, alpha, beta, gamma)
scale_free_graph_name = 'scale_free_graph'
apply_louvain(scale_free_graph)
print('{} has modularity of {}'.format(scale_free_graph_name, compute_modularity(scale_free_graph, use_louvain=True)))
save_graph_as_adj_list(scale_free_graph, scale_free_graph_name)
num_communities = len(set(nx.get_node_attributes(scale_free_graph, 'louvain_community').values()))
plot(scale_free_graph, '{}_louvain_comm'.format(scale_free_graph_name), num_communities, use_louvain=True)


# Create a scale free graph
barabasi_albert_graph = generate_barabasi_albert_graph(100, 1)
barabasi_albert_graph_name = 'barabasi_albert_graph'
apply_louvain(barabasi_albert_graph)
print('{} has modularity of {}'.format(barabasi_albert_graph_name, compute_modularity(barabasi_albert_graph, use_louvain=True)))
save_graph_as_adj_list(barabasi_albert_graph, barabasi_albert_graph_name)
num_communities = len(set(nx.get_node_attributes(barabasi_albert_graph, 'louvain_community').values()))
plot(barabasi_albert_graph, '{}_louvain_comm'.format(barabasi_albert_graph_name), num_communities, use_louvain=True)

