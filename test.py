import io
import os
import random
from PIL import Image
import pygal
from io import BytesIO
import pygraphviz as pgv
import networkx as nx

from minio import Minio

buffer_dict = {}

def plot_graph(graph, num):
    g = pgv.AGraph(directed=False)
    for node, data in graph.nodes(data=True):
        node_color = '#FFFFFF'
        print('**add_node call**')
        print(f'node: {node}')
        print(f"label: <{node}:<b>{data['value']}</b>>")
        g.add_node(node, style='filled', fillcolor=node_color,
                   label=f"<{node}:<b>{data['value']}</b>>"
        )

    # Add edges to the graph
    for edge in graph.edges():
        g.add_edge(edge[0], edge[1])

    # Layout the graph
    g.layout(prog='dot')
    # Draw the graph to a byte buffer
    buffer = io.BytesIO()
    g.draw(buffer, format='svg')
    buffer_dict[f'test{num}'] = buffer






adj_list = "1 2 3,2 3,3 4,4 5,5 6 7,6 7,7 8,8"
g = nx.parse_adjlist(adj_list.split(','))
for i in range(0,10):
    for n in g.nodes:
        g.nodes[n]['value'] = random.randint(0, 10)
    plot_graph(g, i)

# Assume you have an array of SVG bytes buffers named svg_buffers
# And assume you want to create a 2-second animation with 10 frames


# Assume you have an array of SVG bytes buffers named svg_buffers
# And assume you want to create a 2-second animation with 10 frames

# Create the SVG document
chart = pygal.XY(width=800, height=600)

# Define the frames
for i, svg_buffer in buffer_dict.values():
    chart.add(f"Frame {i}", [(0, 0)])
    chart.add(f"Frame {i}", [(1, 1)], show_dots=False, show_legend=False)
    chart.add(f"Frame {i}", [(2, 2)], show_dots=False, show_legend=False)
    chart.add(f"Frame {i}", [(3, 3)], show_dots=False, show_legend=False)
    chart.add(f"Frame {i}", [(4, 4)], show_dots=False, show_legend=False)
    chart.add(f"Frame {i}", [(5, 5)], show_dots=False, show_legend=False)
    chart.add(f"Frame {i}", [(6, 6)], show_dots=False, show_legend=False)
    chart.add(f"Frame {i}", [(7, 7)], show_dots=False, show_legend=False)
    chart.add(f"Frame {i}", [(8, 8)], show_dots=False, show_legend=False)
    chart.add(f"Frame {i}", [(9, 9)], show_dots=False, show_legend=False)

# Add the animation to the document
chart.range = (0, 10)
chart.interpolate = 'cubic'
chart.duration = 2000
chart.dynamic_print_values = True

# Render the chart to a byte buffer
buffer = BytesIO()
chart.render_to_png(buffer)

# Save the buffer to a file
with open('./animation.png', 'wb') as f:
    f.write(buffer.getvalue())

