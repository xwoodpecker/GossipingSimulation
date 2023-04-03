import io
import os
import random
from io import BytesIO
from PIL import Image
import imageio
import pygraphviz as pgv
import networkx as nx

from minio import Minio

buffer_dict = {}
images = []


def plot_graph(graph, num):
    g = pgv.AGraph(directed=False)
    for node, data in graph.nodes(data=True):
        node_color = '#FFFFFF'
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
    g.draw(buffer, format='png')
    buffer_dict[f'test{num}'] = buffer
    img = Image.open(buffer)
    images.append(img)




adj_list = "1 2 3,2 3,3 4,4 5,5 6 7,6 7,7 8,8"
g = nx.parse_adjlist(adj_list.split(','))
for i in range(0, 10):
    for n in g.nodes:
        g.nodes[n]['value'] = random.randint(0, 10)
    plot_graph(g, i)

with io.BytesIO() as output:
    imageio.mimsave(output, images, format='GIF', duration=0.25 * len(buffer_dict.items()))  #

    # Create another BytesIO instance and write the gif_bytes to it
    gif_buffer = io.BytesIO(output.getvalue())

object_name = 'test_visualized'
file_size = len(gif_buffer.getbuffer())
gif_buffer.seek(0)  # Rewind the buffer to the beginning

minio_endpoint = os.environ.get("MINIO_ENDPOINT")
minio_user = os.environ.get("MINIO_USER")
minio_password = os.environ.get("MINIO_PASSWORD")

client = Minio(
        minio_endpoint,
        access_key=minio_user,
        secret_key=minio_password,
        secure=False
    )

client.put_object(
    "simulations",
    object_name,
    gif_buffer,
    file_size,
    content_type="image/gif"
)
print(f"Successfully uploaded '{object_name}' to bucket 'simulations'.")
