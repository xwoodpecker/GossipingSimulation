import concurrent
import itertools
import os
import io
import asyncio
import imageio
from PIL import Image, ImageDraw, ImageFont
import grpc
import time
import datetime
import socket
import sys
import re
import json
import colorsys
import networkx as nx
import pygraphviz as pgv
from minio import Minio
from minio.error import S3Error
import logging

sys.path.append("/app/grpc_compiled")
import gossip_pb2
import gossip_pb2_grpc
from cfg import *

# Create a custom logger
log = logging.getLogger(__name__)

# Set the logging level
log.setLevel(logging.INFO)

# Create a formatter with the desired log message format
formatter = logging.Formatter('%(levelname)s:%(message)s')

# Create a handler and set the formatter
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Add the handler to the logger
log.addHandler(handler)


class MinioAccess:
    """
    Represents access parameters for Minio.
    """

    def __init__(self, endpoint, user, password):
        """
        Initialize Minio access parameters.

        Args:
            endpoint (str): The Minio server endpoint.
            user (str): The user for Minio access.
            password (str): The password for Minio access.
        """
        self.endpoint = endpoint
        self.user = user
        self.password = password


class AlgorithmData:
    """
    Represents data related to an algorithm.
    """

    def __init__(self, name, simulation_properties):
        """
        Initialize algorithm data.

        Args:
            name (str): The name of the algorithm.
            simulation_properties (dict): The properties of the simulation.
        """

        self.name = name
        self.properties = simulation_properties
        self.algorithm_parameters = None

    def set_algorithm_parameters(self, algorithm_parameters):
        """
        set algorithm parameters.

        Args:
            algorithm_parameters (dict): The algorithm parameters.
        """
        self.algorithm_parameters = algorithm_parameters


class GraphData:
    """
    Represents data related to a graph.
    """

    def __init__(self, adj_list, nodes, graph, visualize, graph_properties, node_communities=None):
        """
        Initialize graph data.

        Args:
            adj_list (dict): The adjacency list representation of the graph.
            nodes (list): The list of nodes in the graph.
            graph (networkx.Graph): The graph object.
            visualize (bool): Flag indicating whether to visualize the graph.
            graph_properties (dict): The properties of the graph.
            node_communities (dict, optional): The communities of the nodes. Defaults to None.
        """
        self.adj_list = adj_list
        self.nodes = nodes
        self.graph = graph
        # Regular expression to match the number after the last '-' in the node name
        pattern = re.compile(REGEX_NODE_NAME_PATTERN)
        self.node_dict = dict((node_name, re.findall(pattern, node_name)[-1]) for node_name in nodes)
        self.visualize = visualize
        self.properties = graph_properties
        # set the community field to the respective id if necessary
        if node_communities:
            self.node_communities = node_communities
            for node, community_id in node_communities.items():
                self.graph.nodes[node]['community'] = community_id
            self.community_ids = set(node_communities.values())
            self.num_communities = len(self.community_ids)
            self.properties['nodeCommunities'] = node_communities
        else:
            self.community_ids = set()
            self.num_communities = 0


class Result:
    """
    Represents the result of a simulation.
    """

    def __init__(self, num_rounds, algorithm_data, adj_list, graph_metadata=None):
        """
        Initialize the result of a simulation.

        Args:
            num_rounds (int): The number of rounds in the simulation.
            algorithm_data (AlgorithmData): The data of the algorithm used in the simulation.
            adj_list (dict): The adjacency list representation of the graph.
            graph_metadata (dict, optional): Additional metadata about the graph. Defaults to None.
        """
        self.timestamp = datetime.datetime.now().strftime(RESULT_TIME_FORMAT_STRING)
        self.num_rounds = num_rounds
        self.algorithm = algorithm_data.name
        self.simulation_properties = algorithm_data.properties
        self.adj_list = adj_list
        self.graph_metadata = graph_metadata
        self.dict = {
            'timestamp': self.timestamp,
            'num_rounds': self.num_rounds,
            'algorithm': self.algorithm,
            'adj_list': self.adj_list
        }
        if algorithm_data.algorithm_parameters is not None and algorithm_data.algorithm_parameters:
            for param_name, param_value in algorithm_data.algorithm_parameters.items():
                self.dict[param_name.lower()] = param_value
        if self.graph_metadata is not None and self.graph_metadata:
            self.dict['graph_metadata'] = self.graph_metadata
        if self.simulation_properties is not None and self.simulation_properties:
            self.dict['algorithm_metadata'] = self.simulation_properties

    def to_buffered_reader(self):
        """
        Convert the result to a buffered reader.

        Returns:
            io.BufferedReader: A buffered reader containing the result data.
        """
        return io.BufferedReader(io.BytesIO(str(self).encode()))

    def file_size(self):
        """
        Get the file size of the result data.

        Returns:
            int: The file size in bytes.
        """
        json_data_bytes = str(self).encode()
        return len(json_data_bytes)

    def __str__(self):
        """
        Convert the result to a string representation.

        Returns:
            str: The string representation of the result.
        """
        return json.dumps(self.dict, indent=2)


class NodeValueHistory:
    """
    Represents the history of node values in a simulation.
    """

    def __init__(self):
        """
        Initialize the NodeValueHistory object with an empty history.
        """
        self.history = []

    def add(self, round_num, node, value):
        """
        Add a node value to the history.

        Args:
            round_num (int): The round number.
            node (str): The node identifier.
            value: The value associated with the node.
        """
        round_data = next((data for data in self.history if data['round_num'] == round_num), None)
        if round_data is None:
            round_data = {'round_num': round_num, 'nodes': {node: value}}
            self.history.append(round_data)
        else:
            round_data['nodes'][node] = value

    def to_buffered_reader(self):
        """
        Convert the node value history to a buffered reader.

        Returns:
            io.BufferedReader: A buffered reader containing the node value history data.
        """
        return io.BufferedReader(io.BytesIO(str(self).encode()))

    def file_size(self):
        """
        Get the file size of the node value history data.

        Returns:
            int: The file size in bytes.
        """
        json_data_bytes = str(self).encode()
        return len(json_data_bytes)

    def __str__(self):
        """
        Convert the node value history data to a string representation.

        Returns:
            str: The string representation of the node value history data.
        """
        return json.dumps(self.history, indent=2)


class GossipRunner:
    """
    Represents the runner for the gossip simulation.
    Makes the hosts gossip round by round and stores the simulation results.
    """

    def __init__(self, simulation_name, algorithm_data, repeated, multiple_params, graph_data, minio_access):
        """
        Initialize the GossipRunner.

        Args:
            simulation_name (str): The name of the simulation.
            algorithm_data (AlgorithmData): The data of the algorithm used in the simulation.
            repeated (bool): Flag indicating whether the simulation should be repeated.
            multiple_params (bool): Flag indicating whether the simulation contains multiple param sets.
            graph_data (GraphData): The data of the graph used in the simulation.
            minio_access (MinioAccess): The access credentials for MinIO.
        """
        self.simulation_name = simulation_name
        self.algorithm_data = algorithm_data
        self.algorithm = algorithm_data.name
        # set run number and results if repeated execution
        self.repeated = repeated
        self.multiple_params = multiple_params
        if repeated or multiple_params:
            self.results = []
            if repeated:
                self.run_number = 1
            if multiple_params:
                self.execution_number = 1
        self.graph_data = graph_data
        # shortcut to nodes
        self.nodes = graph_data.nodes
        self.graph = graph_data.graph
        # set the grpc gossip stubs that are used for runner to node communication
        self.stubs = [gossip_pb2_grpc.GossipStub(grpc.insecure_channel(f"{node}:{GRPC_SERVICE_PORT}")) for node in
                      nodes]
        self.stub_dict = {node: stub for node, stub in zip(nodes, self.stubs)}
        self.buffer_dict = {}
        self.minio_access = minio_access
        # init graph plot, generates pgv graph with colors and labels
        # self.init_graph_plot(self.graph_data.num_communities)

    def init_next_run(self):
        """
        Initialize the next run of the simulation by incrementing the run number and resetting the state.
        """
        self.run_number += 1
        self.reset()

    def init_next_execution(self):
        """
        Initialize the next execution of the simulation by incrementing the run number and resetting the state.
        """
        self.execution_number += 1
        self.run_number = 1
        self.reset()
        self.results = []

    def set_algorithm_parameters(self, algorithm_parameters):
        """
        Initialize additional algorithm parameters.

        Args:
            algorithm_parameters (dict): Dictionary containing key-value-pairs of parameters.
        """
        self.algorithm_data.set_algorithm_parameters(algorithm_parameters)

    def reset(self):
        """
        Reset the state of the simulation by invoking the Reset method for each node.
        """
        self.buffer_dict = {}
        for node in self.nodes:
            log.info(f"Invoking Reset for node {node}.")
            response = self.stub_dict[node].Reset(gossip_pb2.ResetRequest())

    def init_graph_plot(self, num_comm=0):
        """
        Initialize the graph plot for visualization.

        Args:
            num_comm (int): The number of communities in the graph (default: 0).
        """

        def generate_colors(num_colors):
            """
            Generate a list of colors for visualization.

            Args:
                num_colors (int): The number of colors to generate.

            Returns:
                List: A list of RGB color tuples.
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
            log.info(f'Generated {num_colors} colors for visualization')
            return colors

        if num_comm > 0:
            colors = generate_colors(num_comm)
            log.info(f'Community colors generated')

        # initialie the pgv graph
        g = pgv.AGraph(directed=False)

        if self.graph.number_of_nodes() <= MAXIMUM_NODE_NUMBER_NORMAL_PLOT:
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
                    node_color = DEFAULT_NODE_COLOR
                g.add_node(node, style='filled', fillcolor=node_color)
                log.info(f'Added node {node} with color {node_color} to pgv graph')
            log.info(f'All nodes set for pgv graph')

            # Add edges to the graph
            for edge in self.graph.edges():
                g.add_edge(edge[0], edge[1])
            log.info(f'All edges added for pgv graph')
        else:
            if num_comm > 0:
                # Add communities as nodes to the new graph
                for i in range(0, num_comm):
                    node_color = '#' + ''.join(format(c, '02x') for c in colors.pop(0))
                    g.add_node(i, style='filled', fillcolor=node_color)

                # Add edges between connected communities
                for edge in graph.edges():
                    node1, node2 = edge
                    community1 = graph.nodes[node1]['community']
                    community2 = graph.nodes[node2]['community']
                    if community1 != community2:
                        g.add_edge(community1, community2)
            else:
                log.info('No communities but too many nodes present, no pgv graph will be set.')
                return
        # set the gpv graph to the local var
        self.pgv_graph = g

    def generate_gexf(self, round_num):
        """
        Create the gephi graph file

        Args:
            round_num (int): The current round number.
        """
        # Draw the graph to a byte buffer
        buffer = io.BytesIO()
        nx.write_gexf(self.graph, buffer)
        self.buffer_dict[round_num] = buffer
        log.info(f'Created gephi file for graph {graph_name} in round {round_num}')

    def plot_graph(self, round_num, num_comm=0):
        """
        Plot the graph to visualize the state of the nodes.
        Updates node labels according to the current round.
        Uses stored pgv graph to minimize compuation load.

        Args:
            round_num (int): The current round number.
            num_comm (int): The number of communities in the graph (default: 0).
        """
        # Draw the graph using Graphviz
        if self.graph.number_of_nodes() <= MAXIMUM_NODE_NUMBER_NORMAL_PLOT:
            for node, data in self.graph.nodes(data=True):
                label = f"<{node}:<b>{data['value']}</b>>"
                self.pgv_graph.get_node(node).attr['label'] = label

        else:
            if num_comm > 0:
                for i in range(0, num_comm):
                    community_nodes = [node for node, community in self.node_communities.items() if community == i]
                    values = []
                    for node in community_nodes:
                        node_value = self.graph.nodes[node]['value']
                        values.append(node_value)
                    avg_value = sum(values) / len(values)
                    label = f"<<b>{round(avg_value, 2)}</b>>"
                    self.pgv_graph.get_node(i).attr['label'] = label
            else:
                log.info('No graph will be rendered when more than 100 nodes are present and no communities exist')
                return

        # Layout the graph
        self.pgv_graph.layout(prog='neato')

        # Draw the graph to a byte buffer
        buffer = io.BytesIO()
        self.pgv_graph.draw(buffer, format='png')
        self.buffer_dict[round_num] = buffer
        log.info(f'Created plot for graph {graph_name} in round {round_num}')

    def update_graph_node(self, node, value):
        """
        Update the value of a node in the graph.

        Args:
            node: The node identifier.
            value: The new value for the node.
        """
        graph_node = self.graph_data.node_dict[node]
        self.graph.nodes[graph_node]['value'] = value

    def init_minio(self):
        """
        Initialize the Minio client and create the 'simulations' bucket if it does not exist.

        Returns:
            Minio: The Minio client instance.
        """
        client = Minio(
            self.minio_access.endpoint,
            access_key=self.minio_access.user,
            secret_key=self.minio_access.password,
            secure=False
        )

        # Make 'simulations' bucket if not exist.
        found = client.bucket_exists(MINIO_BUCKET_NAME)
        if not found:
            client.make_bucket(MINIO_BUCKET_NAME)
        else:
            log.info(f"Bucket '{MINIO_BUCKET_NAME}' already exists")

        return client

    def store_results(self):
        """
        Store the simulation results in Minio.
        """
        try:
            client = self.init_minio()

            current_date = datetime.datetime.now().strftime(MINIO_TIME_FORMAT_STRING)

            self.simulation_path = f'{simulation_name}-{simulation_id}/{graph_name}'
            if self.multiple_params:
                self.simulation_path += f'/exec-{self.execution_number}'
            object_path = self.simulation_path
            if self.repeated:
                object_path += f'/run-{self.run_number}'

            def store_images_gifs():
                images = {}
                max_width = 0
                max_height = 0
                for round_num, buffer in self.buffer_dict.items():
                    object_name = f'round-{round_num}'
                    file_size = buffer.getbuffer().nbytes
                    buffer.seek(0)  # Rewind the buffer to the beginning
                    client.put_object(
                        MINIO_BUCKET_NAME,
                        f'{object_path}/rounds/{object_name}.png',
                        buffer,
                        file_size,
                        content_type="image/png"
                    )
                    log.info(f"Successfully uploaded '{object_name}' to bucket '{MINIO_BUCKET_NAME}'.")
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

                if len(gif_images) > 0:
                    with io.BytesIO() as output:
                        # show one frame for 4 seconds
                        imageio.mimsave(output, gif_images, format='GIF', fps=1 / 4)

                        # Create another BytesIO instance and write the gif_bytes to it
                        gif_buffer = io.BytesIO(output.getvalue())

                    object_name = f'{simulation_name}_visualized'
                    file_size = len(gif_buffer.getbuffer())
                    gif_buffer.seek(0)  # Rewind the buffer to the beginning
                    client.put_object(
                        MINIO_BUCKET_NAME,
                        f'{object_path}/{object_name}.gif',
                        gif_buffer,
                        file_size,
                        content_type="image/gif"
                    )
                    log.info(f"Successfully uploaded '{object_name}' to bucket '{MINIO_BUCKET_NAME}'.")

            def store_gexf_files():
                for round_num, buffer in self.buffer_dict.items():
                    object_name = f'{graph_name}-round-{round_num}'
                    file_size = buffer.getbuffer().nbytes
                    buffer.seek(0)  # Rewind the buffer to the beginning
                    client.put_object(
                        MINIO_BUCKET_NAME,
                        f'{object_path}/rounds/{object_name}.gexf',
                        buffer,
                        file_size,
                        content_type='application/xml'
                    )
                    log.info(f"Successfully uploaded '{object_name}' to bucket '{MINIO_BUCKET_NAME}'.")

            # store_images_gifs()
            store_gexf_files()

            result = Result(self.num_rounds, self.algorithm_data, self.graph_data.adj_list, self.graph_data.properties)
            if self.repeated:
                self.results.append(result)
            object_name = f'{simulation_name}_summary'
            client.put_object(
                MINIO_BUCKET_NAME,
                f'{object_path}/{object_name}.json',
                result.to_buffered_reader(),
                result.file_size(),
                content_type="application/json",
                # metadata={"": ""},
            )
            log.info(f"Successfully uploaded '{object_name}' to bucket '{MINIO_BUCKET_NAME}'.")

            object_name = f'{simulation_name}_node_value_history'
            client.put_object(
                MINIO_BUCKET_NAME,
                f'{object_path}/{object_name}.json',
                self.node_value_history.to_buffered_reader(),
                self.node_value_history.file_size(),
                content_type="application/json",
                # metadata={"": ""},
            )
            log.info(f"Successfully uploaded '{object_name}' to bucket '{MINIO_BUCKET_NAME}'.")

        except S3Error as exc:
            log.error("minio error occurred: ", exc)

    def store_average_results(self):
        """
        Store the averaged simulation results in Minio.
        """
        try:
            client = self.init_minio()

            object_path = self.simulation_path

            avg_num_rounds = sum(result.num_rounds for result in self.results) / len(self.results)
            averaged_result = Result(avg_num_rounds, self.algorithm_data, self.graph_data.adj_list,
                                     self.graph_data.properties)
            object_name = f'{simulation_name}_averaged_result'
            client.put_object(
                MINIO_BUCKET_NAME,
                f'{object_path}/{object_name}.json',
                averaged_result.to_buffered_reader(),
                averaged_result.file_size(),
                content_type="application/json",
                # metadata={"": ""},
            )
        except S3Error as exc:
            log.error("minio error occurred: ", exc)

    def grpc_current_value(self, stub):
        return stub.CurrentValue(gossip_pb2.CurrentValueRequest())

    def grpc_gossip(self, stub):
        return stub.Gossip(gossip_pb2.GossipRequest())

    def grpc_stop_application(self, stub):
        return stub.StopApplication(gossip_pb2.StopApplicationRequest())

    def init_node_value_history(self):
        """
        Initialize the node value history and update the graph with initial values.
        """
        self.node_value_history = NodeValueHistory()

        # Create a ThreadPoolExecutor with the desired number of threads
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.stub_dict))
        # Submit the gRPC calls to the executor
        futures = [executor.submit(self.grpc_current_value, self.stub_dict[node]) for node in self.stub_dict]
        # Wait for all the calls to finish
        concurrent.futures.wait(futures)
        value_responses = []
        for future in futures:
            try:
                result = future.result()
                value_responses.append(result)
            except Exception as e:
                value_responses.append(e)
                log.error(f"Error occurred while executing future: {e}")

        # log.info('Debug: all async current value tasks completed')

        for i, node in enumerate(self.stub_dict):
            response = value_responses[i]
            if isinstance(response, Exception):
                log.error(f"Error receiving current value on node {node}: {response}")
            else:
                value = int(response.value)
                log.info(f"Node {node} has initial value {value}.")
                self.node_value_history.add(0, self.graph_data.node_dict[node], value)
                self.update_graph_node(node, value)
        if self.graph_data.visualize:
            # self.plot_graph(0, self.graph_data.num_communities)
            self.generate_gexf(0)

    def run(self):
        """
        Run the gossip simulation.
        """
        round_num = 1
        while True:
            log.info(f"Starting round {round_num} of gossiping...")
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.stub_dict))

            # Submit the gRPC calls to the executor
            # futures = [executor.submit(self.grpc_gossip, self.stub_dict[node]) for node in self.stub_dict]
            # Wait for all the calls to finish
            # concurrent.futures.wait(futures)

            # better do it synchronously to avoid non-deterministic behaviour
            for node in self.stub_dict:
                self.grpc_gossip(self.stub_dict[node])

            # log.info('Debug: all async gossip tasks completed')
            log.info(f"Round {round_num} of gossiping ended. Printing results.")

            # Submit the gRPC calls to the executor
            futures = [executor.submit(self.grpc_current_value, self.stub_dict[node]) for node in self.stub_dict]
            # Wait for all the calls to finish
            concurrent.futures.wait(futures)
            value_responses = []
            for future in futures:
                try:
                    result = future.result()
                    value_responses.append(result)
                except Exception as e:
                    value_responses.append(e)
                    log.error(f"Error occurred while executing future: {e}")

            # log.info('Debug: all async current value tasks completed')

            values = []
            for i, node in enumerate(self.stub_dict):
                response = value_responses[i]
                if isinstance(response, Exception):
                    log.error(f"Error receiving current value on node {node}: {response}")
                else:
                    value = int(response.value)
                    log.info(f"Node {node} has value {value}.")
                    values.append(value)
                    self.node_value_history.add(round_num, self.graph_data.node_dict[node], value)
                    self.update_graph_node(node, value)
            if self.graph_data.visualize:
                # self.plot_graph(round_num, self.graph_data.num_communities)
                self.generate_gexf(round_num)
            if all(value == values[0] for value in values):
                log.info(f"All hosts have converged on value {values[0]}")
                break
            round_num += 1
        log.info(f"The full value history for this run: {self.node_value_history}")
        self.num_rounds = round_num

    def stop_node_applications(self):
        """
        Stop all node applications over gRPC asynchronously.
        """
        log.info(f"Stopping node applications...")
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.stub_dict))
        # Submit the gRPC calls to the executor
        futures = [executor.submit(self.grpc_stop_application, self.stub_dict[node]) for node in self.stub_dict]
        # Wait for all the calls to finish
        concurrent.futures.wait(futures)
        stop_app_responses = []
        for future in futures:
            try:
                result = future.result()
                stop_app_responses.append(result)
            except Exception as e:
                stop_app_responses.append(e)
                log.error(f"Error occurred while executing future: {e}")

        # log.info('Debug: all async stop application tasks completed')

        for node, response in zip(self.stub_dict, stop_app_responses):
            if isinstance(response, Exception):
                log.error(f"Error stopping application on node {node}: {response}")
            else:
                log.info(f"Successfully stopped application on node {node}")


def create_graph(adj_list):
    """
    Create a graph based on the provided adjacency list.

    Args:
        adj_list (str): The adjacency list representing the graph.

    Returns:
        graph (networkx.Graph): The created graph.
    """
    # Read the adjacency list and create a graph
    log.info(f'Creating graph for adjacency list: {adj_list}')
    graph = nx.parse_adjlist(adj_list.split(','))
    log.info(f'Graph created with nodes: {graph.nodes}')
    return graph


if __name__ == '__main__':

    log.info('Gossip runner started.')
    # minio environment variables
    minio_endpoint = os.environ.get(ENVIRONMENT_MINIO_ENDPOINT)
    minio_user = os.environ.get(ENVIRONMENT_MINIO_USER)
    minio_password = os.environ.get(ENVIRONMENT_MINIO_PASSWORD)
    minioAccess = MinioAccess(minio_endpoint, minio_user, minio_password)
    # general environment variables
    simulation_id = os.environ.get(ENVIRONMENT_SIMULATION_ID)
    simulation_name = os.environ.get(ENVIRONMENT_SIMULATION)
    if simulation_name is None:
        simulation_name = DEFAULT_SIMULATION_NAME
    graph_name = os.environ.get(ENVIRONMENT_GRAPH_NAME)
    if graph_name is None:
        graph_name = DEFAULT_GRAPH_NAME
    log.info(f"Running simulation {simulation_name} with ID {simulation_id} on graph {graph_name}.")
    algorithm = os.environ.get(ENVIRONMENT_ALGORITHM)
    if algorithm is None:
        algorithm = DEFAULT_ALGORITHM
    log.info(f"Using algorithm {algorithm} for neighbor selection.")
    # env for series simulation
    series_simulation_str = os.environ.get(ENVIRONMENT_SERIES_SIMULATION)
    if series_simulation_str is None:
        visualize_str = ''
    if series_simulation_str.lower() == 'false':
        series_simulation = False
    else:
        series_simulation = True
    # node communities specific variable
    node_communities = None
    if algorithm in NODE_COMMUNITIES_SET:
        try:
            node_communities = json.loads(os.environ.get(ENVIRONMENT_NODE_COMMUNITIES))
        except (ValueError, TypeError):
            node_communities = {}
    # env variable for repeated execution
    try:
        repetitions = json.loads(os.environ.get(ENVIRONMENT_REPETITIONS))
    except (ValueError, TypeError):
        repetitions = DEFAULT_REPETITIONS
    repeated = False
    if repetitions > 1:
        repeated = True
        log.info(f'Simulation will be repeated for a total of {repetitions} runs.')
    # load simulation properties for logging
    try:
        simulation_properties = json.loads(os.environ.get(ENVIRONMENT_SIMULATION_PROPERTIES))
    except (ValueError, TypeError):
        simulation_properties = {}
    # initialize algorithm data
    algorithmData = AlgorithmData(algorithm, simulation_properties)
    # also load graph properties for logging
    try:
        graph_properties = json.loads(os.environ.get(ENVIRONMENT_GRAPH_PROPERTIES))
    except (ValueError, TypeError):
        graph_properties = {}
    # generate the graph from the adj list
    adj_list = os.environ.get(ENVIRONMENT_ADJ_LIST).rstrip(',')
    graph = create_graph(adj_list)
    # get and sort the network nodes for communication
    nodes = os.environ.get(ENVIRONMENT_NODES).rstrip(',').split(",")

    algorithm_params = dict()
    factors = []
    if algorithm in WEIGHTED_FACTOR_SET:
        try:
            factors = os.environ.get(ENVIRONMENT_FACTOR).rstrip(',').split(",")
        except (ValueError, AttributeError):
            factors = [DEFAULT_FACTOR]
        algorithm_params[ENVIRONMENT_FACTOR] = factors

    prior_partner_factors = []
    if algorithm in MEMORY_SET:
        try:
            prior_partner_factors = os.environ.get(ENVIRONMENT_PRIOR_PARTNER_FACTOR).rstrip(',').split(",")
        except (ValueError, AttributeError):
            prior_partner_factors = [DEFAULT_PRIOR_PARTNER_FACTOR]
        algorithm_params[ENVIRONMENT_PRIOR_PARTNER_FACTOR] = prior_partner_factors

    algorithm_param_keys = list(algorithm_params.keys())
    algorithm_param_values = list(algorithm_params.values())

    algorithm_param_combinations = [dict(zip(algorithm_param_keys, combo))
                                    for combo in itertools.product(*algorithm_param_values) if combo]


    def natural_sort_key(full_name):
        # Extract the numeric part of the name using regular expressions
        match = re.findall(r'\d+', full_name)
        if match:
            # Convert the last numeric part to an actual number for sorting
            return int(match[-1])
        else:
            # If the name does not match the expected pattern, treat it as a high value
            return float('inf')


    # nodes = sorted(nodes, key=natural_sort_key)

    log.info(f"Received network nodes: {nodes}")
    # env for visualization settings
    visualize_str = os.environ.get(ENVIRONMENT_VISUALIZE)
    if visualize_str is None:
        visualize_str = ''
    if visualize_str.lower() == 'false':
        visualize = False
    else:
        visualize = True
    # initialize graph data
    graphData = GraphData(adj_list, nodes, graph, visualize, graph_properties, node_communities)

    no_params = False
    num_executions = len(algorithm_param_combinations)
    if num_executions == 0:
        no_params = True
        num_executions = 1

    multi_params = False
    if num_executions > 1:
        multi_params = True

    # initialize the gossip runner
    runner = GossipRunner(simulation_name, algorithmData, repeated, multi_params, graphData, minioAccess)

    for i in range(0, num_executions):
        sep = '*' * 50
        log.info(f"{sep}")
        log.info(f'Execution #{i + 1}/{num_executions} starting...')
        if not no_params:
            chosen_params = algorithm_param_combinations[i]
            log.info(f"Set of params: {chosen_params}")
            runner.set_algorithm_parameters(chosen_params)
        # repeat for additional repetitions
        for j in range(0, repetitions):
            sep = '-' * 50
            log.info(f"{sep}")
            log.info(f'Run #{j + 1}/{repetitions} executing...')
            runner.init_node_value_history()
            runner.run()
            runner.store_results()
            if j < repetitions - 1:
                runner.init_next_run()
        if repeated:
            # store averaged results (over multiple runs)
            runner.store_average_results()

        if i < num_executions - 1:
            runner.init_next_execution()

    # stop all nodes over grpc
    runner.stop_node_applications()
    log.info("Stopping application.")
    sys.exit(0)
