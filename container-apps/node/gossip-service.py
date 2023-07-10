import itertools

import grpc
import random
import time
import sys
import socket
import os
from concurrent import futures
import threading
import json
import copy
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


class ValueEntry:
    """
    Represents a value entry with the number of gossiping participations 
    and the current value.
    """

    def __init__(self, participations, value):
        """
        Initialize a ValueEntry object.

        Args:
            participations (int): The number of participations.
            value (int): The value.
        """
        self.participations = int(participations)
        self.value = int(value)


class Algorithm:
    """
    Represents an algorithm with a name and a list of neighbors.
    The algorithm's function is to select a neighbor for gossiping.
    """

    def __init__(self, name, neighbors):
        """
        Initialize an Algorithm object.

        Args:
            name (str): The name of the algorithm.
            neighbors (list): The list of neighbors.
        """
        self.name = name
        self.neighbors = neighbors
        log.info(f"Running algorithm: {self.name}.")
        log.info(f"Received neighbors: {self.neighbors}.")
        self.modifiable_parameters = False

    def select_neighbor(self):
        """
        Select a neighbor randomly.

        Returns:
            object: The selected neighbor.
        """
        return random.choice(self.neighbors)


class Memory:
    """
    Represents a memory object with memory storage and prior partner factor.
    The factor is used to discourage repeated gossiping between the same nodes.
    """

    def init_memory(self, prior_partner_factors):
        """
        Initialize the memory.

        Args:
            prior_partner_factors (list): The prior partner factors.
        """
        self.memory = set()
        self.prior_partner_factors = prior_partner_factors
        self.prior_partner_factor_index = 0
        self.prior_partner_factor = self.prior_partner_factors[self.prior_partner_factor_index]
        self.modifiable_parameters = True
        log.info(f'Initialized Memory with prior partner factor {self.prior_partner_factor}.')

    def remember(self, neighbor):
        """
        Remember a neighbor.

        Args:
            neighbor: The neighbor to remember.
        """
        if neighbor not in self.memory:
            self.weights[neighbor] *= self.prior_partner_factor
            self.memory.add(neighbor)
            log.info('DEBUG: weights got changed!!')
        log.info(f'DEBUG: weights after gossip {self.weights.values()}')

    def reset_memory(self):
        """
        Reset the memory to the start config.
        """
        log.info('Memory reset.')
        self.memory = set()

    def set_next_memory_parameters(self):
        """
        Sets the next memory parameters from the list.
        """
        self.prior_partner_factor_index += 1
        self.prior_partner_factor = self.prior_partner_factors[self.prior_partner_factor_index]
        log.info(f'Set new prior partner factor {self.prior_partner_factor}.')


class ComplexMemory(Memory):
    """
    Represents a complex memory object that inherits from Memory.
    Each interaction with a neighbor further penalizes the probability for selection.
    It also enhances the normal memory with the ability to forget.
    Forgetting restores the start weights.
    """

    def init_memory(self, prior_partner_factors):
        """
        Initialize the complex memory.

        Args:
            prior_partner_factors (list): The prior partner factors.
        """
        self.prior_partner_factors = prior_partner_factors
        self.prior_partner_factor_index = 0
        self.prior_partner_factor = self.prior_partner_factors[self.prior_partner_factor_index]
        self.start_weights = copy.deepcopy(self.weights)
        self.modifiable_parameters = True
        log.info(f'Initialized ComplexMemory with prior partner factor {self.prior_partner_factor}.')

    def remember(self, neighbor):
        """
        Remember a neighbor and update weights.

        Args:
            neighbor: The neighbor to remember.
        """
        self.weights[neighbor] *= self.prior_partner_factor
        log.info(f'DEBUG: weights after gossip {self.weights.values()}')

    def forget(self):
        """
        Forget previously remembered neighbors and reset weights.
        """
        self.weights = copy.deepcopy(self.start_weights)
        log.info(f'DEBUG: weights after forgetting {self.weights.values()}')

    def reset_memory(self):
        """
        Reset the memory to the start config.
        """
        log.info('Memory reset.')
        self.forget()


class DefaultMemory(Algorithm, Memory):
    """
    Represents the default memory algorithm. It inherits from both Algorithm and Memory.
    """

    def __init__(self, name, neighbors, prior_partner_factors):
        """
        Initialize the default memory.

        Args:
            name (str): The name of the memory.
            neighbors (list): The list of neighbors.
            prior_partner_factors (list): The prior partner factors.
        """
        super().__init__(name, neighbors)
        self.weights = {}
        for neighbor in self.neighbors:
            self.weights[neighbor] = 1.0
        super().init_memory(prior_partner_factors)

    def select_neighbor(self):
        """
        Select a neighbor based on weights.

        Returns:
            object: The selected neighbor.
        """
        selected = random.choices(self.neighbors, weights=self.weights.values())[0]
        return selected

    def set_next_parameters(self):
        """
        Sets the next parameters from the list.
        """
        self.set_next_memory_parameters()


class DefaultComplexMemory(Algorithm, ComplexMemory):
    """
    Represents the default complex memory algorithm. It inherits from both Algorithm and ComplexMemory.
    """

    def __init__(self, name, neighbors, prior_partner_factors):
        """
        Initialize the default complex memory.

        Args:
            name (str): The name of the memory.
            neighbors (list): The list of neighbors.
            prior_partner_factors (list): The prior partner factors.
        """
        super().__init__(name, neighbors)
        self.weights = {}
        for neighbor in self.neighbors:
            self.weights[neighbor] = 1.0
        super().init_memory(prior_partner_factors)

    def select_neighbor(self):
        """
        Select a neighbor based on weights.

        Returns:
            object: The selected neighbor.
        """
        selected = random.choices(self.neighbors, weights=self.weights.values())[0]
        return selected

    def set_next_parameters(self):
        """
        Sets the next parameters from the list.
        """
        self.set_next_memory_parameters()


class WeightedFactor(Algorithm):
    """
    Represents a weighted factor object that inherits from Algorithm.
    """

    def __init__(self, name, neighbors, community_neighbors, factors):
        """
        Initialize the weighted factor.

        Args:
            name (str): The name of the weighted factor.
            neighbors (list): The list of neighbors.
            community_neighbors (list): The list of community neighbors.
            factors (list): The factor to prioritize non-community neighbors.
        """
        super().__init__(name, neighbors)
        self.community_neighbors = community_neighbors
        self.factors = factors
        self.factor_index = 0
        self.factor = self.factors[self.factor_index]
        self.compute_weights()
        self.modifiable_parameters = True
        log.info(f'Initialized WeightedFactor with factor {self.factor}.')

    def compute_weights(self):
        self.weights = {}
        for neighbor in self.neighbors:
            if neighbor in self.community_neighbors:
                weight = 1.0
            else:
                # prioritize non-community neighbors with a given factor
                weight = self.factor
            self.weights[neighbor] = weight

    def select_neighbor(self):
        """
        Select a neighbor based on weights.

        Returns:
            object: The selected neighbor.
        """
        selected = random.choices(self.neighbors, weights=self.weights.values())[0]
        return selected

    def set_next_parameters(self):
        """
        Sets the next parameters from the list.
        """
        self.factor_index += 1
        self.factor = self.factors[self.factor_index]
        log.info(f'Set new factor {self.factor}.')
        self.compute_weights()


class WeightedFactorMemory(WeightedFactor, Memory):
    """
    Represents the weighted factor memory algorithm that inherits from both WeightedFactor and Memory.
    """

    def __init__(self, name, neighbors, community_neighbors, factors, prior_partner_factors):
        """
        Initialize the weighted factor memory.

        Args:
            name (str): The name of the memory.
            neighbors (list): The list of neighbors.
            community_neighbors (list): The list of community neighbors.
            factors (list): The factors to prioritize non-community neighbors.
            prior_partner_factors (list): The prior partner factors.
        """
        super().__init__(name, neighbors, community_neighbors, factors)
        super().init_memory(prior_partner_factors)

    def set_next_parameters(self):
        """
        Sets the next parameters from the list.
        """
        super().set_next_memory_parameters()
        super().set_next_parameters()


class WeightedFactorComplexMemory(WeightedFactor, ComplexMemory):
    """
    Represents the weighted factor complex memory algorithm that inherits from both WeightedFactor and ComplexMemory.
    """

    def __init__(self, name, neighbors, community_neighbors, factors, prior_partner_factors):
        """
        Initialize the weighted factor complex memory.

        Args:
            name (str): The name of the memory.
            neighbors (list): The list of neighbors.
            community_neighbors (list): The list of community neighbors.
            factors (list): The factors to prioritize non-community neighbors.
            prior_partner_factors (list): The prior partner factors.
        """
        super().__init__(name, neighbors, community_neighbors, factors)
        super().init_memory(prior_partner_factors)

    def set_next_parameters(self):
        """
        Sets the next parameters from the list.
        """
        super().set_next_memory_parameters()
        super().set_next_parameters()


class CommunityProbabilities(Algorithm):
    """
    Represents a community probabilities object that inherits from Algorithm.
    """

    def __init__(self, name, neighbors, same_community_probabilities_neighbors):
        """
        Initialize the community probabilities.

        Args:
            name (str): The name of the community probabilities.
            neighbors (list): The list of neighbors.
            same_community_probabilities_neighbors (list): The list of same community probabilities for neighbors.
        """
        super().__init__(name, neighbors)
        self.same_community_probabilities_neighbors = same_community_probabilities_neighbors
        inverted_probabilities = [1 / x for x in same_community_probabilities_neighbors]
        # set weights to inverted probabilities
        self.weights = {}
        for i in range(0, len(self.neighbors)):
            self.weights[neighbors[i]] = inverted_probabilities[i]
        log.info(f'DEBUG: weights at the start {self.weights.values()}')

    def select_neighbor(self):
        """
        Select a neighbor based on weights.

        Returns:
            object: The selected neighbor.
        """
        selected = random.choices(self.neighbors, weights=self.weights.values())[0]
        return selected


class CommunityProbabilitiesMemory(CommunityProbabilities, Memory):
    """
    Represents a community probabilities memory algorithm that inherits from both CommunityProbabilities and Memory.
    """

    def __init__(self, name, neighbors, same_community_probabilities_neighbors, prior_partner_factors):
        """
        Initialize the community probabilities memory.

        Args:
            name (str): The name of the memory.
            neighbors (list): The list of neighbors.
            same_community_probabilities_neighbors (list): The list of same community probabilities for neighbors.
            prior_partner_factors (list): The prior partner factors.
        """
        super().__init__(name, neighbors, same_community_probabilities_neighbors)
        super().init_memory(prior_partner_factors)


    def set_next_parameters(self):
        """
        Sets the next parameters from the list.
        """
        super().set_next_memory_parameters()


class CommunityProbabilitiesComplexMemory(CommunityProbabilities, ComplexMemory):
    """
    Represents a community probabilities complex memory algorithm that inherits from both CommunityProbabilities and ComplexMemory.
    """

    def __init__(self, name, neighbors, same_community_probabilities_neighbors, prior_partner_factors):
        """
        Initialize the community probabilities complex memory.

        Args:
            name (str): The name of the memory.
            neighbors (list): The list of neighbors.
            same_community_probabilities_neighbors (list): The list of same community probabilities for neighbors.
            prior_partner_factors (list): The prior partner factors.
        """
        super().__init__(name, neighbors, same_community_probabilities_neighbors)
        super().init_memory(prior_partner_factors)

    def set_next_parameters(self):
        """
        Sets the next parameters from the list.
        """
        super().set_next_memory_parameters()


class GossipService(gossip_pb2_grpc.GossipServicer):
    """
    Represents a Gossip service that implements the GossipServicer interface.
    """

    def __init__(self, name, algorithm, node_value, stop_event):
        """
        Initialize the Gossip service.

        Args:
            name (str): The name of the Gossip service.
            algorithm (Algorithm): The algorithm used for gossip.
            node_value (ValueEntry): The value entry of the node.
            stop_event (threading.Event): The event used to stop the service.
        """
        self.name = name
        self.algorithm = algorithm
        self._stop_event = stop_event
        self.stop_listening = False
        # assign the node value
        if node_value is None:
            self.value = random.randint(0, 100)
        else:
            self.value = int(node_value)
        self.original_value = self.value
        self.participations = 0
        # set the value entries and its inivial entry
        self.value_entries = []
        self.value_entries.append(ValueEntry(0, self.value))
        log.info(f'GossipService initialized on {self.name} with value {self.value}.')
        self.repetitions_counter = 0
        # Start the listen_for_connections method in a background thread
        self.listen_thread = threading.Thread(target=self.listen_for_connections)
        self.listen_thread.start()

    def process_gossip_data(self, data):
        """
        Process the received gossip data.

        Args:
            data (bytes): The received gossip data.

        Returns:
            None
        """
        self.participations += 1
        peer_name, peer_value = data.decode('utf-8').split(':')
        log.info(f"Received peer value: {peer_value} from {peer_name}.")
        thread_id = threading.get_ident()
        log.info(f"DEBUG {thread_id} - Received peer value: {peer_value} from {peer_name}")
        log.info(f"DEBUG {thread_id} - Current value: {self.value}")
        old_value = self.value
        # set the minimum as the new value
        self.value = min(self.value, int(peer_value))
        if old_value == self.value:
            log.info(f"Value was not changed. Current value {self.value}")
        else:
            log.info(f"Value was changed. New value after gossiping: {self.value}.")
            if isinstance(self.algorithm, ComplexMemory):
                self.algorithm.forget()
        # log the participation as a value entry
        self.value_entries.append(ValueEntry(self.participations, self.value))
        # remember the neighbor in case it is a memory algorithm
        if isinstance(self.algorithm, Memory):
            self.algorithm.remember(peer_name)

    def gossip(self):
        """
        Perform the gossip operation.

        Returns:
            None
        """
        log.info('Starting to gossip.')
        neighbor = self.algorithm.select_neighbor()
        log.info(f'Selecting neighbor {neighbor} to gossip using algorithm {self.algorithm.name}.')
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.connect((neighbor, TCP_SERVICE_PORT))
                client_socket.sendall(bytes(f'{self.name}:{self.value}', 'utf-8'))
                log.info(f"Sent {self.value} over TCP socket.")
                data = client_socket.recv(TCP_BUFSIZE)
                # process the received data
                self.process_gossip_data(data)
        except socket.error as e:
            log.error(f"Socket error occurred: {str(e)}")
        log.info('Finished gossiping.')

    def Reset(self, request, context):
        """
        Handle the GRPC Reset request.

        Args:
            request (gossip_pb2.ResetRequest): The Reset request.
            context (grpc._server._Context): The context of the request.

        Returns:
            gossip_pb2.ResetResponse: The Reset response.
        """
        log.info('[GRPC Reset invoked]')
        self.value = self.original_value
        self.participations = 0
        self.value_entries = []
        self.value_entries.append(ValueEntry(0, self.value))
        sep = '-' * 50
        log.info(f"{sep}")
        log.info(f'GossipService reset on {self.name} with original value {self.value}.')
        self.repetitions_counter += 1
        if self.algorithm.name in MEMORY_SET:
            self.algorithm.reset_memory()
        log.info(f'DEBUG : {self.repetitions_counter} mod {repetitions} = {self.repetitions_counter % repetitions}')
        if self.algorithm.modifiable_parameters and self.repetitions_counter % repetitions == 0:
            self.algorithm.set_next_parameters()

        return gossip_pb2.ResetResponse()

    def Gossip(self, request, context):
        """
        Handle the GRPC Gossip request.

        Args:
            request (gossip_pb2.GossipRequest): The Gossip request.
            context (grpc._server._Context): The context of the request.

        Returns:
            gossip_pb2.GossipResponse: The Gossip response.
        """
        log.info('[GRPC Gossip invoked]')
        self.gossip()
        return gossip_pb2.GossipResponse()

    def History(self, request, context):
        """
        Handle the GRPC History request.

        Args:
            request (gossip_pb2.HistoryRequest): The History request.
            context (grpc._server._Context): The context of the request.

        Returns:
            gossip_pb2.HistoryResponse: The History response.
        """
        log.info('[GRPC History invoked]')
        value_entries = [gossip_pb2.ValueEntry(participations=entry.participations, value=entry.value)
                         for entry in self.value_entries]
        return gossip_pb2.HistoryResponse(value_entries=value_entries)

    def CurrentValue(self, request, context):
        """
        Handle the GRPC CurrentValue request.

        Args:
            request (gossip_pb2.CurrentValueRequest): The CurrentValue request.
            context (grpc._server._Context): The context of the request.

        Returns:
            gossip_pb2.CurrentValueResponse: The CurrentValue response.
        """
        log.info('[GRPC CurrentValue invoked]')
        log.info(f"Returning value {self.value}.")
        return gossip_pb2.CurrentValueResponse(value=self.value)

    def StopApplication(self, request, context):
        """
        Handle the GRPC StopApplication request.

        Args:
            request (gossip_pb2.StopApplicationRequest): The StopApplication request.
            context (grpc._server._Context): The context of the request.

        Returns:
            gossip_pb2.StopApplicationResponse: The StopApplication response.
        """
        log.info('[GRPC StopApplication invoked]')
        self.stop_listening = True
        self.listen_thread.join()
        log.info("Stopped listening.")
        self._stop_event.set()
        return gossip_pb2.StopApplicationResponse()

    def handle_connection(self, conn):
        """
        Handle an incoming gossip connection.

        Args:
            conn (socket.socket): The socket connection.

        Returns:
            None
        """
        try:
            # Process incoming data
            data = conn.recv(TCP_BUFSIZE)
            self.process_gossip_data(data)
            conn.sendall(bytes(f'{self.name}:{self.value}', 'utf-8'))
            log.info(f"Responded with value of {self.value} over TCP socket.")
        finally:
            conn.close()

    def listen_for_connections(self):
        """
        Start listening for incoming gossip connections.

        Returns:
           None
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.bind((self.name, TCP_SERVICE_PORT))
                server_socket.listen(1)
                log.info(f'Listening on port {TCP_SERVICE_PORT} for incoming gossip...')
                while not self.stop_listening:
                    server_socket.settimeout(3)
                    try:
                        conn, addr = server_socket.accept()
                        log.info(f'Gossiping request accepted from {addr[0]}:{addr[1]}')
                        # Handle the connection in a separate thread
                        conn_thread = threading.Thread(target=self.handle_connection, args=(conn,))
                        conn_thread.start()
                    except socket.timeout:
                        # log.error('Socket Timeout occurred.')
                        continue
        except socket.error as e:
            log.error(f"Socket error occurred: {str(e)}")


if __name__ == '__main__':

    log.info('Gossip node service started.')

    # get general needed data from environment variables
    name = os.environ.get(ENVIRONMENT_HOSTNAME)
    neighbors = os.environ.get(ENVIRONMENT_NEIGHBORS).rstrip(',').split(",")
    algorithm_name = os.environ.get(ENVIRONMENT_ALGORITHM)
    if algorithm_name is None:
        algorithm_name = DEFAULT_ALGORITHM

    # env variable for repeated execution
    try:
        repetitions = json.loads(os.environ.get(ENVIRONMENT_REPETITIONS))
    except (ValueError, TypeError):
        repetitions = DEFAULT_REPETITIONS

    algorithm_params = dict()
    factors = []
    if algorithm_name in WEIGHTED_FACTOR_SET:
        try:
            factors = os.environ.get(ENVIRONMENT_FACTOR).rstrip(',').split(",")
            factors = [float(factor) for factor in factors]
        except (ValueError, AttributeError):
            factors = [DEFAULT_FACTOR]
        algorithm_params[ENVIRONMENT_FACTOR] = factors

    prior_partner_factors = []
    if algorithm_name in MEMORY_SET:
        try:
            prior_partner_factors = os.environ.get(ENVIRONMENT_PRIOR_PARTNER_FACTOR).rstrip(',').split(",")
            prior_partner_factors = [float(factor) for factor in prior_partner_factors]
        except (ValueError, AttributeError):
            prior_partner_factors = [DEFAULT_PRIOR_PARTNER_FACTOR]
        algorithm_params[ENVIRONMENT_PRIOR_PARTNER_FACTOR] = prior_partner_factors

    algorithm_param_keys = list(algorithm_params.keys())
    algorithm_param_values = list(algorithm_params.values())

    algorithm_param_combinations = [dict(zip(algorithm_param_keys, combo))
                                    for combo in itertools.product(*algorithm_param_values) if combo]

    # check for random initialization and node value assignment
    nodeValue = None
    randomInitialization_str = os.environ.get(ENVIRONMENT_RANDOM_INITIALIZATION)
    if randomInitialization_str is None:
        randomInitialization_str = ''
    if randomInitialization_str.lower() == 'false':
        randomInitialization = False
    else:
        randomInitialization = True
    if not randomInitialization:
        nodeValue = os.environ.get(ENVIRONMENT_NODE_VALUE)


    def init_default_algorithm():
        """
        Initialize the default algorithm.

        Returns:
            Algorithm: The default algorithm.
        """
        return Algorithm(DEFAULT_ALGORITHM, neighbors)


    def init_communities_and_factors():
        """
        Initialize the community neighbors and factors.

        Returns:
            tuple: A tuple containing the community neighbors and factors.
        """
        community_neighbors = os.environ.get(ENVIRONMENT_COMMUNITY_NEIGHBORS).rstrip(',').split(",")
        factors = [item[ENVIRONMENT_FACTOR] for item in algorithm_param_combinations if ENVIRONMENT_FACTOR in item]
        log.info(f'Community neighbors set to {community_neighbors}')
        log.info(f'Factors set to {factors}')
        return community_neighbors, factors


    def init_same_community_probabilities_neighbors():
        """
        Initialize the same community probabilities neighbors.

        Returns:
            list: The same community probabilities neighbors.
        """
        same_community_probabilities_neighbors_str = os.environ.get(ENVIRONMENT_SAME_COMMUNITY_PROBABILITIES_NEIGHBORS)
        same_community_probabilities_neighbors = [float(item) for item in
                                                  same_community_probabilities_neighbors_str.rstrip(',').split(",")]
        log.info(f'Same community probabilities set to {same_community_probabilities_neighbors}')
        return same_community_probabilities_neighbors


    def init_memory():
        """
        Initialize the prior partner factors.

        Returns:
            list: The prior partner factors.
        """
        prior_partner_factors = [item[ENVIRONMENT_PRIOR_PARTNER_FACTOR] for item in algorithm_param_combinations
                                 if ENVIRONMENT_PRIOR_PARTNER_FACTOR in item]
        log.info(f'Prior partner factors set to {prior_partner_factors}')
        return prior_partner_factors


    def init_default_memory():
        """
        Initialize the default memory algorithm.

        Returns:
            DefaultMemory: The default memory.
        """
        prior_partner_factors = init_memory()
        return DefaultMemory(algorithm_name, neighbors, prior_partner_factors)


    def init_default_complex_memory():
        """
        Initialize the default complex memory algorithm.

        Returns:
            DefaultComplexMemory: The default complex memory.
        """
        prior_partner_factors = init_memory()
        return DefaultComplexMemory(algorithm_name, neighbors, prior_partner_factors)


    def init_weighted_factor():
        """
        Initialize the weighted factor algorithm.

        Returns:
            WeightedFactor: The weighted factor.
        """
        community_neighbors, factors = init_communities_and_factors()
        return WeightedFactor(algorithm_name, neighbors, community_neighbors, factors)


    def init_weighted_factor_memory():
        """
        Initialize the weighted factor memory algorithm.

        Returns:
            WeightedFactorMemory: The weighted factor memory.
        """
        community_neighbors, factors = init_communities_and_factors()
        prior_partner_factors = init_memory()
        return WeightedFactorMemory(algorithm_name, neighbors, community_neighbors, factors, prior_partner_factors)


    def init_weighted_factor_complex_memory():
        """
        Initialize the weighted factor complex memory algorithm.

        Returns:
            WeightedFactorComplexMemory: The weighted factor complex memory.
        """
        community_neighbors, factors = init_communities_and_factors()
        prior_partner_factors = init_memory()
        return WeightedFactorComplexMemory(algorithm_name, neighbors, community_neighbors, factors,
                                           prior_partner_factors)


    def init_community_probabilities():
        """
        Initialize the community probabilities algorithm.

        Returns:
            CommunityProbabilities: The community probabilities.
        """
        same_community_probabilities_neighbors = init_same_community_probabilities_neighbors()
        return CommunityProbabilities(algorithm_name, neighbors, same_community_probabilities_neighbors)


    def init_community_probabilities_memory():
        """
        Initialize the community probabilities memory algorithm.

        Returns:
            CommunityProbabilitiesMemory: The community probabilities memory.
        """
        same_community_probabilities_neighbors = init_same_community_probabilities_neighbors()
        prior_partner_factors = init_memory()
        return CommunityProbabilitiesMemory(algorithm_name, neighbors, same_community_probabilities_neighbors,
                                            prior_partner_factors)


    def init_community_probabilities_complex_memory():
        """
        Initialize the community probabilities complex memory algorithm.

        Returns:
            CommunityProbabilitiesComplexMemory: The community probabilities complex memory.
        """
        same_community_probabilities_neighbors = init_same_community_probabilities_neighbors()
        prior_partner_factors = init_memory()
        return CommunityProbabilitiesComplexMemory(algorithm_name, neighbors, same_community_probabilities_neighbors,
                                                   prior_partner_factors)


    def init_algorithm(name):
        """
        Initialize the algorithm based on the given name.

        Args:
            name (str): The name of the algorithm.

        Returns:
            Algorithm: The initialized algorithm.
        """
        init_funcs = {
            ALGORITHM_DEFAULT_MEMORY: init_default_memory,
            ALGORITHM_DEFAULT_COMPLEX_MEMORY: init_default_complex_memory,
            ALGORITHM_WEIGHTED_FACTOR: init_weighted_factor,
            ALGORITHM_WEIGHTED_FACTOR_MEMORY: init_weighted_factor_memory,
            ALGORITHM_WEIGHTED_FACTOR_COMPLEX_MEMORY: init_weighted_factor_complex_memory,
            ALGORITHM_COMMUNITY_PROBABILITIES: init_community_probabilities,
            ALGORITHM_COMMUNITY_PROBABILITIES_MEMORY: init_community_probabilities_memory,
            ALGORITHM_COMMUNITY_PROBABILITIES_COMPLEX_MEMORY: init_community_probabilities_complex_memory
        }
        init_func = init_funcs.get(name, init_default_algorithm)
        algorithm = init_func()
        return algorithm


    # set stop event
    stop_event = threading.Event()
    # init service
    service = GossipService(name, init_algorithm(algorithm_name), nodeValue, stop_event)
    # init grpc server
    server = grpc.server(futures.ThreadPoolExecutor())
    gossip_pb2_grpc.add_GossipServicer_to_server(service, server)
    server.add_insecure_port(f'[::]:{GRPC_SERVICE_PORT}')
    # start the server
    server.start()
    log.info(f"Server started on port {GRPC_SERVICE_PORT}")
    # Wait for the stop event to be set when stop application grpc call is invoked
    stop_event.wait()

    # Application is stopped over grpc
    log.info("Stopping server...")
    server.stop(0)
    log.info("Server stopped.")
    log.info("Stopping application.")
    sys.exit(0)
