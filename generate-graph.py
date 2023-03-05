import networkx as nx
import json
import matplotlib.pyplot as plt

def save_graph_as_json_and_plot():
    # Create a new graph
    G = nx.Graph()

    # Add nodes and edges to the graph
    G.add_nodes_from([1, 2, 3, 4])
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])

   # Save the graph as an adjacency list
    with open('graph.adjlist', 'w') as f:
        for line in nx.generate_adjlist(G):
            f.write(line + ',')

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos)

    # Save the plot as a PNG file
    plt.savefig('graph.png')

# Call the function to create and save the graph
save_graph_as_json_and_plot()
