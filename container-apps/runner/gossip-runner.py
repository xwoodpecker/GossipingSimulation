import os
import io
import imageio
from PIL import Image, ImageDraw, ImageFont
import grpc
import time
import datetime
import socket
import  sys
import re
import json
import colorsys
import networkx as nx

import pygraphviz as pgv

from minio import Minio
from minio.error import S3Error

sys.path.append("/app/grpc_compiled")
import gossip_pb2
import gossip_pb2_grpc

class MinioAccess:
    def __init__(self, endpoint, user, password):
        self.endpoint = endpoint
        self.user = user
        self.password = password

class GraphData:
    def __init__(self, nodes, graph, visualize, graph_properties):
        self.nodes = nodes
        self.graph = graph
         # Regular expression to match the number after the last '-' in the node name
        pattern = re.compile(r'-(\d+)(?![\d-])')
        self.node_dict = dict((node_name, re.findall(pattern, node_name)[-1]) for node_name in nodes)
        self.visualize = visualize
        self.properties = graph_properties

class GossipRunner:
    def __init__(self, simulation_name, algorithm, graph_data, minio_access):
        self.simulation_name = simulation_name
        self.algorithm = algorithm
        self.graph_data = graph_data
        # shortcut to nodes
        self.nodes = graph_data.nodes
        self.graph = graph_data.graph
        self.stubs = [gossip_pb2_grpc.GossipStub(grpc.insecure_channel(f"{node}:50051")) for node in nodes]
        self.stub_dict = {node: stub for node, stub in zip(nodes, self.stubs)}
        self.buffer_dict = {}
        #print(f"Stubs set to {self.stub_dict}")
        self.minio_access = minio_access

    def plot_graph(self, round_num, num_comm=0):

        def generate_colors(num_colors):
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
        
        # Draw the graph using Graphviz
        g = pgv.AGraph(directed=False)

        if num_comm > 0:
            colors = generate_colors(num_comm)

        # Assign a color to each node based on its community
        community_colors = {}
        for node, data in self.graph.nodes(data=True):
            if num_comm > 0:
                community_id = data['community']
                if community_id not in community_colors:
                    community_colors[community_id] = colors.pop(0)
                node_color = '#' + ''.join(format(c, '02x') for c in community_colors[community_id])
            else:
                node_color = '#FFFFFF'
            label=f"<{node}:<b>{data['value']}</b>>"
            g.add_node(node, style='filled', fillcolor=node_color, label=label)

        # Add edges to the graph
        for edge in self.graph.edges():
            g.add_edge(edge[0], edge[1])

        # Layout the graph
        g.layout(prog='dot')

        # Draw the graph to a byte buffer
        buffer = io.BytesIO()
        g.draw(buffer, format='png')
        self.buffer_dict[round_num] = buffer
        print(f'Created plot for simulation {simulation_name} in round {round_num}')

    def update_graph_node(self, node, value):
            graph_node = self.graph_data.node_dict[node]
            self.graph.nodes[graph_node]['value'] = value

    
    def store_results(self):

        class Result:
            def __init__(self, num_rounds, algorithm, adj_list, metadata=None):
                self.num_rounds = num_rounds
                self.algorithm = algorithm
                self.adj_list = adj_list
                self.metadata = metadata
                self.buffered_reader = None
                dict =  {
                    'num_rounds': self.num_rounds,
                    'algorithm': self.algorithm,
                    'adj_list': self.adj_list
                }
                if self.metadata is not None and self.metadata:
                    dict['metadata'] = self.metadata
                self.json_data = json.dumps(dict)
                        
            def to_buffered_reader(self):
                if self.buffered_reader is None:
                    self.buffered_reader = io.BufferedReader(io.BytesIO(self.json_data.encode()))
                return self.buffered_reader
            
            def file_size(self):
                json_data_bytes = self.json_data.encode()
                return len(json_data_bytes)
        
        try:
            client = Minio(
                self.minio_access.endpoint,
                access_key=self.minio_access.user,
                secret_key=self.minio_access.password,
                secure=False
            )

            # Make 'simulations' bucket if not exist.
            found = client.bucket_exists("simulations")
            if not found:
                client.make_bucket("simulations")
            else:
                print("Bucket 'simulations' already exists")

            current_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
            object_path=f'{simulation_name}-{current_date}'

            images = []
            for round_num, buffer in self.buffer_dict.items():
                object_name = f'{simulation_name}-round-{round_num}'
                file_size = buffer.getbuffer().nbytes
                buffer.seek(0)  # Rewind the buffer to the beginning
                client.put_object(
                    "simulations",
                    f'{object_path}/rounds/{object_name}.png',
                    buffer,
                    file_size,
                    content_type="image/png"
                )
                print(f"Successfully uploaded '{object_name}' to bucket 'simulations'.")
                img = Image.open(buffer)
                title = f"Round#{round_num}"
                # Create a new image with enough space for the original image and the text below it
                new_image = Image.new('RGB', (img.width, img.height + 50), color=(255, 255, 255))
                # Paste the original image into the new image
                new_image.paste(img, (0, 0))
                # Draw the text at the bottom of the new image
                draw = ImageDraw.Draw(new_image)
                text_bbox = draw.textbbox((0, img.height, new_image.width, new_image.height), title)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = (new_image.width - text_width) / 2
                text_y = img.height + 10
                draw.text((text_x, text_y), title, fill=(0, 0, 0))
                images.append(new_image)

            with io.BytesIO() as output:
                imageio.mimsave(output, images, format='GIF', duration=0.5*len(self.buffer_dict.items()))

                # Create another BytesIO instance and write the gif_bytes to it
                gif_buffer = io.BytesIO(output.getvalue())
            
            object_name=f'{simulation_name}_visualized'
            file_size = len(gif_buffer.getbuffer())
            gif_buffer.seek(0)  # Rewind the buffer to the beginning
            client.put_object(
                "simulations",
                f'{object_path}/{object_name}.gif',
                gif_buffer,
                file_size,
                content_type="image/gif"
            )
            print(f"Successfully uploaded '{object_name}' to bucket 'simulations'.")

            result = Result(self.num_rounds, 'default', adj_list, self.graph_data.properties)
            object_name=f'{simulation_name}_summary'
            client.put_object(
                "simulations",
                f'{object_path}/{object_name}.json',
                result.to_buffered_reader(),
                result.file_size(),
                content_type="application/json",
                # metadata={"": ""},
            )
            print(f"Successfully uploaded '{object_name}' to bucket 'simulations'.")
        except S3Error as exc:
            print("minio error occurred: ", exc)


    def init_value_history(self):
        self.value_history = {}
        for node in self.stub_dict: 
            response = self.stub_dict[node].CurrentValue(gossip_pb2.CurrentValueRequest())
            value = int(response.value)
            self.value_history[node] = [(0, value)]
            self.update_graph_node(node, value)
        if self.graph_data.visualize:
            self.plot_graph(0)

    def run(self):
        round_num = 1
        while True:
            print(f"Starting round {round_num} of gossiping...")
            for node in self.stub_dict:
                print(f"Invoking Gossiping for node {node}.")
                response = self.stub_dict[node].Gossip(gossip_pb2.GossipRequest())
                time.sleep(1)
            print(f"Round {round_num} of gossiping ended. Printing results.")
            values = []
            for node in self.stub_dict:
                response = self.stub_dict[node].CurrentValue(gossip_pb2.CurrentValueRequest())
                value = int(response.value)
                values.append(value)
                print(f"Node {node} has value {value}.")
                self.value_history[node].append((round_num, value))
                self.update_graph_node(node, value)
            if self.graph_data.visualize:
                self.plot_graph(round_num)
            if all(value == values[0] for value in values):
                print(f"All hosts have converged on value {values[0]}")
                break
            round_num += 1
            time.sleep(1)
        print(f"The full value history for this run: {self.value_history}")
        self.num_rounds = round_num

    def stop_node_applications(self):
        print(f"Stopping node applications...")
        for node in self.stub_dict: 
            response = self.stub_dict[node].StopApplication(gossip_pb2.StopApplicationRequest())
            print(f"Sent stop application request to node {node}")

def create_graph(adj_list):
    # Read the adjacency list from the file and create a graph
    print(f'Creating graph for adjacency list: {adj_list}')
    graph = nx.parse_adjlist(adj_list.split(','))
    print(f'Graph created with nodes: {graph.nodes}')
    return graph


if __name__ == '__main__':

    print('Gossip runner started.')
    minio_endpoint = os.environ.get("MINIO_ENDPOINT")
    minio_user = os.environ.get("MINIO_USER")
    minio_password = os.environ.get("MINIO_PASSWORD")

    minioAccess = MinioAccess(minio_endpoint, minio_user, minio_password)

    simulation_name = os.environ.get("SIMULATION")
    if simulation_name is None:
        simulation_name = 'unidentified'

    algorithm = os.environ.get("ALGORITHM")
    if algorithm is None:
        algorithm = 'default'

    # todo: simulation properties (?)
    
    try:
        graph_properties = json.loads(os.environ.get("GRAPH_PROPERTIES"))
    except ValueError:
        graph_properties = {}

    adj_list = os.environ.get("ADJ_LIST").rstrip(',')
    graph = create_graph(adj_list)

    nodes = os.environ.get("NODES").rstrip(',').split(",")
    print(f"Received network nodes: {nodes}")

    visualize = os.environ.get("VISUALIZE")
    if visualize is None or not isinstance(visualize, bool):
        visualize = True

    graphData = GraphData(nodes, graph, visualize, graph_properties)

    runner = GossipRunner(simulation_name, algorithm, graphData, minioAccess)
    runner.init_value_history()
    runner.run()
    runner.store_results()
    runner.stop_node_applications()
    print("Stopping application.")
    sys.exit(0)