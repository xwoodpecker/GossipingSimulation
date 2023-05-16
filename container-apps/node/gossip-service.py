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

sys.path.append("/app/grpc_compiled")
import gossip_pb2
import gossip_pb2_grpc
from cfg import *


class ValueEntry:
    def __init__(self, participations, value):
        self.participations = int(participations)
        self.value = int(value)

class Algorithm:
    def __init__(self, name, neighbors):
        self.name = name
        self.neighbors = neighbors
        print(f"Running algorithm: {self.name}.")
        print(f"Received neighbors: {self.neighbors}.")
    
    def select_neighbor(self):
        return random.choice(self.neighbors)
    
class Memory:
    def init_memory(self, prior_partner_factor):
        self.memory = set()
        self.prior_partner_factor = prior_partner_factor
        
    def remember(self, neighbor):
        if neighbor not in self.memory:
            self.weights[neighbor] *= self.prior_partner_factor
            self.memory.add(neighbor)
            print('DEBUG: weights got changed!!')
        print(f'DEBUG: weights after gossip {self.weights.values()}')

class ComplexMemory(Memory):
    def init_memory(self, prior_partner_factor):
        self.prior_partner_factor = prior_partner_factor
        self.start_weights = copy.deepcopy(self.weights)
        
    def remember(self, neighbor):
        self.weights[neighbor] *= self.prior_partner_factor 
        print(f'DEBUG: weights after gossip {self.weights.values()}')

    def forget(self):
        self.weights = copy.deepcopy(self.start_weights)
        print(f'DEBUG: weights after forgetting {self.weights.values()}')

class DefaultMemory(Algorithm, Memory):
    def __init__(self, name, neighbors, prior_partner_factor):
        super().__init__(name, neighbors)
        self.weights = {}
        for neighbor in self.neighbors:
            self.weights[neighbor] = 1.0
        super().init_memory(prior_partner_factor)
    
    def select_neighbor(self):
        selected = random.choices(self.neighbors, weights=self.weights.values())[0]
        return selected

class DefaultComplexMemory(Algorithm, ComplexMemory):
    def __init__(self, name, neighbors, prior_partner_factor):
        super().__init__(name, neighbors)
        self.weights = {}
        for neighbor in self.neighbors:
            self.weights[neighbor] = 1.0
        super().init_memory(prior_partner_factor)
    
    def select_neighbor(self):
        selected = random.choices(self.neighbors, weights=self.weights.values())[0]
        return selected

class WeightedFactor(Algorithm):
    def __init__(self, name, neighbors, community_neighbors, factor):
        super().__init__(name, neighbors)
        self.community_neighbors = community_neighbors
        self.factor = factor

        self.weights = {}
        for neighbor in self.neighbors:
            if neighbor in self.community_neighbors:
                weight = 1.0
            else:
                # prioritize non-community neighbors with a given factor
                weight = self.factor
            self.weights[neighbor] = weight
        print(f'DEBUG: weights at the start {self.weights.values()}')
    
    def select_neighbor(self):
        selected = random.choices(self.neighbors, weights=self.weights.values())[0]
        return selected

class WeightedFactorMemory(WeightedFactor, Memory):
    def __init__(self,  name, neighbors, community_neighbors, factor, prior_partner_factor):
        super().__init__(name, neighbors, community_neighbors, factor)
        super().init_memory(prior_partner_factor) 
        
class WeightedFactorComplexMemory(WeightedFactor, ComplexMemory):
    def __init__(self,  name, neighbors, community_neighbors, factor, prior_partner_factor):
        super().__init__(name, neighbors, community_neighbors, factor)
        super().init_memory(prior_partner_factor)  

        
class CommunityProbabilities(Algorithm):
    def __init__(self, name, neighbors, same_community_probabilities_neighbors):
        super().__init__(name, neighbors)
        self.same_community_probabilities_neighbors = same_community_probabilities_neighbors
        inverted_probabilities = [1 / x for x in same_community_probabilities_neighbors]
        # set weights to inverted probabilities
        self.weights = {}
        for i in range(0, len(self.neighbors)):
            self.weights[neighbors[i]] = inverted_probabilities[i]
        print(f'DEBUG: weights at the start {self.weights.values()}')

    def select_neighbor(self):
        selected = random.choices(self.neighbors, weights=self.weights.values())[0]
        return selected

class CommunityProbabilitiesMemory(CommunityProbabilities, Memory):
    def __init__(self, name, neighbors, same_community_probabilities_neighbors, prior_partner_factor):
        super().__init__(name, neighbors, same_community_probabilities_neighbors)
        super().init_memory(prior_partner_factor)

class CommunityProbabilitiesComplexMemory(CommunityProbabilities, ComplexMemory):
    def __init__(self, name, neighbors, same_community_probabilities_neighbors, prior_partner_factor):
        super().__init__(name, neighbors, same_community_probabilities_neighbors)
        super().init_memory(prior_partner_factor)

        

class GossipService(gossip_pb2_grpc.GossipServicer):
    def __init__(self, name, algorithm, node_value, stop_event):
        self.name = name
        self.algorithm = algorithm
        self._stop_event = stop_event
        self.stop_listening = False
        if node_value is None:
            self.value = random.randint(0, 100)
        else:
            self.value = int(node_value)
        self.original_value = self.value
        self.participations = 0
        self.value_entries = []
        self.value_entries.append(ValueEntry(0, self.value))
        print(f'GossipService initialized on {self.name} with value {self.value}.')
        # Start the listen_for_connections method in a background thread
        self.listen_thread = threading.Thread(target=self.listen_for_connections)
        self.listen_thread.start()

    def process_gossip_data(self, data):
        self.participations += 1
        peer_name, peer_value = data.decode('utf-8').split(':')
        print(f"Received peer value: {peer_value} from {peer_name}")
        print(f"Current value: {self.value}")
        old_value = self.value
        self.value = min(self.value, int(peer_value))
        if old_value == self.value:
            print(f"Value was not changed")
        else:
            print(f"New value after gossiping: {self.value}")
            if isinstance(self.algorithm, ComplexMemory):
                self.algorithm.forget()
        self.value_entries.append(ValueEntry(self.participations, self.value))
        if isinstance(self.algorithm, Memory):
            self.algorithm.remember(peer_name)


    def gossip(self):
        print('Starting to gossip.')
        neighbor = self.algorithm.select_neighbor()
        print(f'Selecting neighbor {neighbor} to gossip using algorithm {self.algorithm.name}.')
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.connect((neighbor, TCP_SERVICE_PORT))
                client_socket.sendall(bytes(f'{self.name}:{self.value}', 'utf-8'))
                print(f"Sent {self.value} over TCP socket.")
                data = client_socket.recv(TCP_BUFSIZE)
                self.process_gossip_data(data)
        except socket.error as e:
            print(f"Socket error occurred: {str(e)}")
        print('Finished gossiping.')
    
    def Reset(self, request, context):
        print('[GRPC Reset invoked]')
        self.value = self.original_value
        self.participations = 0
        self.value_entries = []
        self.value_entries.append(ValueEntry(0, self.value))
        print(f'GossipService reset on {self.name} with original value {self.value}.')
        return gossip_pb2.ResetResponse()

    def Gossip(self, request, context):
        print('[GRPC Gossip invoked]')
        self.gossip()
        return gossip_pb2.GossipResponse()

    def History(self, request, context):
        print('[GRPC History invoked]')
        value_entries =[gossip_pb2.ValueEntry( participations=entry.participations, value=entry.value) 
            for entry in self.value_entries]
        return gossip_pb2.HistoryResponse(value_entries=value_entries)

    def CurrentValue(self, request, context):
        print('[GRPC CurrentValue invoked]')
        print(f"Returning value {self.value}.")
        return gossip_pb2.CurrentValueResponse(value=self.value)
    
    def StopApplication(self, request, context):
        print('[GRPC StopApplication invoked]')
        self.stop_listening = True
        self.listen_thread.join()
        print("Stopped listening.")
        self._stop_event.set()
        return gossip_pb2.StopApplicationResponse()

    def listen_for_connections(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.bind((self.name, TCP_SERVICE_PORT))
                server_socket.listen(1)
                print(f'Listening on port {TCP_SERVICE_PORT} for incoming gossip...')
                while not self.stop_listening:
                    server_socket.settimeout(3) 
                    try: 
                        conn, addr = server_socket.accept()
                    except socket.timeout:
                        continue 
                    try:
                        print(f'Gossiping request accepted from {addr[0]}:{addr[1]}')
                        # Process incoming data
                        data = conn.recv(TCP_BUFSIZE)
                        self.process_gossip_data(data)
                        conn.sendall(bytes(f'{self.name}:{self.value}', 'utf-8'))
                        print(f"Responded with value of {self.value} over TCP socket.")
                    finally:
                        conn.close()
        except socket.error as e:
            print(f"Socket error occurred: {str(e)}")


if __name__ == '__main__':

    print('Gossip node service started.')
    name = os.environ.get(ENVIRONMENT_HOSTNAME)

    neighbors = os.environ.get(ENVIRONMENT_NEIGHBORS).rstrip(',').split(",")

    algorithm_name = os.environ.get(ENVIRONMENT_ALGORITHM)
    if algorithm_name is None:
        algorithm_name = DEFAULT_ALGORITHM

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
        return Algorithm(DEFAULT_ALGORITHM, neighbors)

    def init_communities_and_factor():
        community_neighbors = os.environ.get(ENVIRONMENT_COMMUNITY_NEIGHBORS).rstrip(',').split(",")
        try:
            factor = float(os.environ.get(ENVIRONMENT_FACTOR))
        except ValueError:
            factor = DEFAULT_FACTOR
        print(f'Community neighbors set to {community_neighbors}')
        print(f'Factor set to {factor}')
        return community_neighbors, factor
    
    def init_same_community_probabilities_neighbors():
        same_community_probabilities_neighbors_str = os.environ.get(ENVIRONMENT_SAME_COMMUNITY_PROBABILITIES_NEIGHBORS)
        same_community_probabilities_neighbors = [float(item) for item in same_community_probabilities_neighbors_str.rstrip(',').split(",")]
        print(f'Same community probabilities set to {same_community_probabilities_neighbors}')
        return same_community_probabilities_neighbors
    
    def init_memory():
        try:
            prior_partner_factor = float(os.environ.get(ENVIRONMENT_PRIOR_PARTNER_FACTOR))
        except ValueError:
            prior_partner_factor = DEFAULT_ALGORITHM
        print(f'Prior partner factor set to {prior_partner_factor}')
        return prior_partner_factor
    
    def init_default_memory():
        prior_partner_factor = init_memory()
        return DefaultMemory(algorithm_name, neighbors, prior_partner_factor)
    
    def init_default_complex_memory():
        prior_partner_factor = init_memory()
        return DefaultComplexMemory(algorithm_name, neighbors, prior_partner_factor)

    def init_weighted_factor():
        community_neighbors, factor = init_communities_and_factor()
        return WeightedFactor(algorithm_name, neighbors, community_neighbors, factor)
    
    def init_weighted_factor_memory():
        community_neighbors, factor = init_communities_and_factor()
        prior_partner_factor = init_memory()
        return WeightedFactorMemory(algorithm_name, neighbors, community_neighbors, factor, prior_partner_factor) 
    
    def init_weighted_factor_complex_memory():
        community_neighbors, factor = init_communities_and_factor()
        prior_partner_factor = init_memory()
        return WeightedFactorComplexMemory(algorithm_name, neighbors, community_neighbors, factor, prior_partner_factor) 
    
    def init_community_probabilities():
        same_community_probabilities_neighbors = init_same_community_probabilities_neighbors()
        return CommunityProbabilities(algorithm_name, neighbors, same_community_probabilities_neighbors)
    
    def init_community_probabilities_memory():
        same_community_probabilities_neighbors = init_same_community_probabilities_neighbors()
        prior_partner_factor = init_memory()
        return CommunityProbabilitiesMemory(algorithm_name, neighbors, same_community_probabilities_neighbors, prior_partner_factor)
    
    def init_community_probabilities_complex_memory():
        same_community_probabilities_neighbors = init_same_community_probabilities_neighbors()
        prior_partner_factor = init_memory()
        return CommunityProbabilitiesComplexMemory(algorithm_name, neighbors, same_community_probabilities_neighbors, prior_partner_factor)

    def init_algorithm(name):
        init_funcs = {
            ALGORITHM_DEFAULT_MEMORY: init_default_memory,
            ALGORITHM_DEFAULT_COMPLEX_MEMORY: init_default_complex_memory,
            ALGORITHM_WEIGHTED_FACTOR : init_weighted_factor,
            ALGORITHM_WEIGHTED_FACTOR_MEMORY: init_weighted_factor_memory,
            ALGORITHM_WEIGHTED_FACTOR_COMPLEX_MEMORY: init_weighted_factor_complex_memory,
            ALGORITHM_COMMUNITY_PROBABILITIES: init_community_probabilities,
            ALGORITHM_COMMUNITY_PROBABILITIES_MEMORY :  init_community_probabilities_memory,
            ALGORITHM_COMMUNITY_PROBABILITIES_COMPLEX_MEMORY :  init_community_probabilities_complex_memory
        }
        init_func = init_funcs.get(name, init_default_algorithm)
        algorithm = init_func()
        return algorithm

    stop_event = threading.Event()
    service = GossipService(name, init_algorithm(algorithm_name), nodeValue, stop_event)
    server = grpc.server(futures.ThreadPoolExecutor())
    gossip_pb2_grpc.add_GossipServicer_to_server(service, server)
    server.add_insecure_port(f'[::]:{GRPC_SERVICE_PORT}')
    server.start()
    print(f"Server started on port {GRPC_SERVICE_PORT}")
    # Wait for the stop event to be set when stop application grpc call is invoked
    stop_event.wait()
    print("Stopping server...")
    server.stop(0)
    print("Server stopped.")
    print("Stopping application.")
    sys.exit(0)