import io
import os
import random
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import imageio
import pygraphviz as pgv
import networkx as nx

from minio import Minio

buffer_dict = {}
images = []


def plot_graph(graph, num):
    g = pgv.AGraph(name=str(num), directed=False)
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

    # Add title to the graph image
    title = f"Round#{num}"

    # Define the font and size for the text
    font = ImageFont.truetype('arial.ttf', size=18)

    # Create a new image with enough space for the original image and the text below it
    new_image = Image.new('RGB', (img.width, img.height + 50), color=(255, 255, 255))

    # Paste the original image into the new image
    new_image.paste(img, (0, 0))

    # Draw the text at the bottom of the new image
    draw = ImageDraw.Draw(new_image)
    text_bbox = draw.textbbox((0, img.height, new_image.width, new_image.height), title, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (new_image.width - text_width) / 2
    text_y = img.height + 10
    draw.text((text_x, text_y), title, font=font, fill=(0, 0, 0))

    images.append(new_image)




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
        "192.168.178.58:32650",
        access_key="admin",
        secret_key="ULeZ4zcYI9",
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
