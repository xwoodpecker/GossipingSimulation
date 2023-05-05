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
    def __init__(self, adj_list, nodes, graph, visualize, graph_properties, node_communities=None):
        self.adj_list = adj_list
        self.nodes = nodes
        self.graph = graph
        # Regular expression to match the number after the last '-' in the node name
        pattern = re.compile(r'-(\d+)(?![\d-])')
        self.node_dict = dict((node_name, re.findall(pattern, node_name)[-1]) for node_name in nodes)
        self.visualize = visualize
        self.properties = graph_properties
        # set the community field to the respective id if necessary
        if node_communities:
            for node, community_id in node_communities.items():
                self.graph.nodes[node]['community'] = community_id
            self.community_ids = set(node_communities.values())
            self.num_communities = len(self.community_ids)
        else:
            self.community_ids = set()
            self.num_communities = 0


class Result:
    def __init__(self, num_rounds, algorithm, adj_list, metadata=None):
        self.num_rounds = num_rounds
        self.algorithm = algorithm
        self.adj_list = adj_list
        self.metadata = metadata
        self.buffered_reader = None
        dict = {
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

class NodeValueHistory:
    def __init__(self):
        self.dict = {}

    def add(self, round_num, node, value):
        self.dict['round_num'] = round_num
        self.dict['round_num'][node] = value
    
    def to_buffered_reader(self):
        if self.json_data is None:
            self.json_data = json.dumps(dict)
        if self.buffered_reader is None:
            self.buffered_reader = io.BufferedReader(io.BytesIO(self.json_data.encode()))
        return self.buffered_reader
    
    def file_size(self):
        json_data_bytes = self.json_data.encode()
        return len(json_data_bytes)

class GossipRunner:
    def __init__(self, simulation_name, algorithm, repeated, graph_data, minio_access):
        self.simulation_name = simulation_name
        self.algorithm = algorithm
        self.repeated = repeated
        if repeated:
            self.results = []
            self.run_number = 1
        self.graph_data = graph_data
        # shortcut to nodes
        self.nodes = graph_data.nodes
        self.graph = graph_data.graph
        self.stubs = [gossip_pb2_grpc.GossipStub(grpc.insecure_channel(f"{node}:50051")) for node in nodes]
        self.stub_dict = {node: stub for node, stub in zip(nodes, self.stubs)}
        self.buffer_dict = {}
        # print(f"Stubs set to {self.stub_dict}")
        self.minio_access = minio_access
        self.init_plot_colors(self.graph_data.num_communities)

    def reset(self):
        self.buffer_dict = {}
        for node in self.nodes:
            print(f"Invoking Reset for node {node}.")
            response = self.stub_dict[node].Reset(gossip_pb2.ResetRequest())

    def init_plot_colors(self, num_comm=0):

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
            print(f'Generated {num_colors} colors for visualization')
            return colors
        
        if num_comm > 0:
            colors = generate_colors(num_comm)

        # Assign a color to each node based on its community
        idx = 0    
        community_colors = {}
        for node, data in self.graph.nodes(data=True):
            if num_comm > 0:
                community_id = data['community']
                if community_id not in community_colors:
                    community_colors[community_id] = colors[idx]
                    idx += 1
                node_color = '#' + ''.join(format(c, '02x') for c in community_colors[community_id])
            else:
                node_color = '#FFFFFF'
            self.graph.nodes[node]['node_color'] = node_color
            print(f'Set node color {node_color} for node {node}')
        print(f'Node colors set for each node in graph')


    def plot_graph(self, round_num, num_comm=0):
        # Draw the graph using Graphviz
        g = pgv.AGraph(directed=False)

        for node, data in self.graph.nodes(data=True):
            label=f"<{node}:<b>{data['value']}</b>>"
            g.add_node(node, style='filled', fillcolor=self.graph.nodes[node]['node_color'], label=label)

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
            
            
            if self.repeated:
                if self.run_number == 1:
                    self.simulation_path = object_path=f'{simulation_name}-{current_date}'
                object_path = self.simulation_path + f'/run-{self.run_number}'
            else:
                object_path=f'{simulation_name}-{current_date}'

            images = {}
            max_width = 0
            max_height = 0
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
                images[round_num] = img
                # find  the largest image
                max_width = max(max_width, img.width)
                max_height = max(max_height, img.height)
            
            gif_images = []
            for round_num, image in images.items():
                title = f"Round#{round_num}"
                # Create a new image with enough space for the original image and the text below it
                new_image = Image.new('RGB', (max_width, max_height + 50), color=(255, 255, 255))
                # Paste the original image into the new image
                new_image.paste(image, (0, 0))
                # Draw the text at the bottom of the new image
                draw = ImageDraw.Draw(new_image)
                text_bbox = draw.textbbox((0, image.height, new_image.width, new_image.height), title)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = (new_image.width - text_width) / 2
                text_y = image.height + 10
                draw.text((text_x, text_y), title, fill=(0, 0, 0))
                gif_images.append(new_image)

            with io.BytesIO() as output:
                imageio.mimsave(output, gif_images, format='GIF', duration=0.5*len(self.buffer_dict.items()))

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

            result = Result(self.num_rounds, self.algorithm, self.graph_data.adj_list, self.graph_data.properties)
            self.results.append(result)
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
            
            object_name=f'{simulation_name}_node_value_history'
            client.put_object(
                "simulations",
                f'{object_path}/{object_name}.json',
                self.node_value_history.to_buffered_reader(),
                self.node_value_history.file_size(),
                content_type="application/json",
                # metadata={"": ""},
            )
            print(f"Successfully uploaded '{object_name}' to bucket 'simulations'.")

        except S3Error as exc:
            print("minio error occurred: ", exc)
        
    def store_average_results(self):
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

            object_path=self.simulation_path

            avg_num_rounds = avg_num_rounds = sum(result.num_rounds for result in self.results) / len(self.results)
            averaged_result = Result(avg_num_rounds, self.algorithm, self.graph_data.adj_list, self.graph_data.properties)
            object_name=f'{simulation_name}_averaged_result'
            client.put_object(
                "simulations",
                f'{object_path}/{object_name}.json',
                averaged_result.to_buffered_reader(),
                averaged_result.file_size(),
                content_type="application/json",
                # metadata={"": ""},
            )
        except S3Error as exc:
            print("minio error occurred: ", exc)


    def init_node_value_history(self):
        self.node_value_history = NodeValueHistory()
        for node in self.stub_dict: 
            response = self.stub_dict[node].CurrentValue(gossip_pb2.CurrentValueRequest())
            value = int(response.value)
            self.node_value_history.add(0, node, value)
            self.update_graph_node(node, value)
        if self.graph_data.visualize:
            self.plot_graph(0, self.graph_data.num_communities)

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
                self.node_value_history.add(0, node, value)
                self.update_graph_node(node, value)
            if self.graph_data.visualize:
                self.plot_graph(round_num, self.graph_data.num_communities)
            if all(value == values[0] for value in values):
                print(f"All hosts have converged on value {values[0]}")
                break
            round_num += 1
            time.sleep(1)
        print(f"The full value history for this run: {self.node_value_history}")
        self.num_rounds = round_num

    def stop_node_applications(self):
        print(f"Stopping node applications...")
        for node in self.stub_dict: 
            response = self.stub_dict[node].StopApplication(gossip_pb2.StopApplicationRequest())
            print(f"Sent stop application request to node {node}")

def create_graph(adj_list):
    # Read the adjacency list and create a graph
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

    node_communities = None
    if algorithm == 'weighted_v0':
        try:
            node_communities = json.loads(os.environ.get("NODE_COMMUNITIES"))
        except (ValueError, TypeError):
            node_communities = {}

    try:
        repititions = json.loads(os.environ.get("REPITITIONS"))
    except (ValueError, TypeError):
        repititions = 1
    repeated = False
    if repititions > 1:
        repeated = True

    # todo: simulation properties (?)
    
    try:
        graph_properties = json.loads(os.environ.get("GRAPH_PROPERTIES"))
    except (ValueError, TypeError):
        graph_properties = {}

    adj_list = os.environ.get("ADJ_LIST").rstrip(',')
    graph = create_graph(adj_list)

    nodes = os.environ.get("NODES").rstrip(',').split(",")
    print(f"Received network nodes: {nodes}")

    try:
        visualize = json.loads(os.environ.get("VISUALIZE"))
    except (ValueError, TypeError):
        visualize = None
    if visualize is None or not isinstance(visualize, bool):
        visualize = True

    graphData = GraphData(adj_list, nodes, graph, visualize, graph_properties, node_communities)

    runner = GossipRunner(simulation_name, algorithm, repeated, graphData, minioAccess)
    runner.init_node_value_history()
    runner.run()
    runner.store_results()

    # repeat for additional repetitions
    for i in range(1, repititions):
        runner.reset()
        runner.init_node_value_history()
        runner.run()
        runner.store_results()
    if repeated:
        runner.store_average_results()

    runner.stop_node_applications()
    print("Stopping application.")
    sys.exit(0)