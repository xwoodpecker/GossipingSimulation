import grpc
import random
import time
import sys
import socket
import os
from concurrent import futures
import threading
import json

sys.path.append("/app/grpc_compiled")
import gossip_pb2
import gossip_pb2_grpc


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

class WeightedFactor(Algorithm):
    def __init__(self, name, neighbors, community_neighbors, factor):
        super().__init__(name, neighbors)
        self.community_neighbors = community_neighbors
        self.factor = factor
    
    def select_neighbor(self):
        weights = []
        for neighbor in self.neighbors:
            if neighbor in self.community_neighbors:
                weight = 1.0
            else:
                # prioritize non-community neighbors with a given factor
                weight = self.algorithm.factor 
            weights.append(weight)
        selected = random.choices(self.neighbors, weights=weights)[0]
        return selected

class WeightedFactorMemory(WeightedFactor):
    def __init__(self,  name, neighbors, community_neighbors, factor, prior_partner_factor):
        super().__init__(name, neighbors, community_neighbors, factor)
        self.prior_partner_factor = prior_partner_factor
        self.memory = set()

        def select_neighbor(self):            
            weights = []
            for neighbor in self.neighbors:
                if neighbor in self.community_neighbors:
                    weight = 1.0
                else:
                    # prioritize non-community neighbors with a given factor
                    weight = self.factor
                if neighbor in self.memory:
                    weight *= self.prior_partner_factor
                weights.append(weight)
            
            selected = random.choices(self.neighbors, weights=weights)[0]
            self.memory.add(selected)  # Remember the selected neighbor
            return selected

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
        peer_value = int(data.decode('utf-8'))
        print(f"Received peer value: {peer_value}, Current value: {self.value}")
        self.value = min(self.value, peer_value)
        print(f"Value after gossiping: {self.value}")
        self.value_entries.append(ValueEntry(self.participations, self.value))


    def gossip(self):
        print('Starting to gossip.')
        neighbor = self.algorithm.select_neighbor()
        print(f'Selecting neighbor {neighbor} to gossip using algorithm {self.algorithm.name}.')
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.connect((neighbor, 90))
                client_socket.sendall(bytes(str(self.value), 'utf-8'))
                print(f"Sent {self.value} over TCP socket.")
                data = client_socket.recv(1024)
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
                server_socket.bind((self.name, 90))
                server_socket.listen(1)
                print('Listening on port 90 for incoming gossip...')
                while not self.stop_listening:
                    server_socket.settimeout(3) 
                    try: 
                        conn, addr = server_socket.accept()
                    except socket.timeout:
                        continue 
                    try:
                        print(f'Gossiping request accepted from {addr[0]}:{addr[1]}')
                        # Process incoming data
                        data = conn.recv(1024)
                        self.process_gossip_data(data)
                        conn.sendall(bytes(str(self.value), 'utf-8'))
                        print(f"Responded with value of {self.value} over TCP socket.")
                    finally:
                        conn.close()
        except socket.error as e:
            print(f"Socket error occurred: {str(e)}")


if __name__ == '__main__':

    print('Gossip node service started.')
    name = os.environ.get("HOSTNAME")

    neighbors = os.environ.get("NEIGHBORS").rstrip(',').split(",")

    algorithm_name = os.environ.get("ALGORITHM")
    if algorithm_name is None:
        algorithm_name = 'default'

    nodeValue = None
    randomInitialization_str = os.environ.get("RANDOM_INITIALIZATION")
    if randomInitialization_str is None:
        randomInitialization_str = ''
    if randomInitialization_str.lower() == 'false':
        randomInitialization = False
    else:
        randomInitialization = True
    if not randomInitialization:
        nodeValue = os.environ.get("NODE_VALUE")

    def init_default_algorithm():
        return Algorithm(algorithm_name, neighbors)

    def init_communities_and_factor():
        community_neighbors = os.environ.get("COMMUNITY_NEIGHBORS").rstrip(',').split(",")
        #non_community_neighbors = os.environ.get("NON_COMMUNITY_NEIGHBORS").rstrip(',').split(",")
        try:
            factor = float(os.environ.get("FACTOR"))
        except ValueError:
            factor = 1.5
        return community_neighbors, factor

    def init_weighted_factor():
        community_neighbors, factor = init_communities_and_factor()
        return WeightedFactor(algorithm_name, neighbors, community_neighbors, factor)
    
    def init_weighted_factor_memory():
        community_neighbors, factor = init_communities_and_factor()
        try:
            prior_partner_factor = float(os.environ.get("PRIOR_PARTNER_FACTOR"))
        except ValueError:
            prior_partner_factor = 0.5
        return WeightedFactorMemory(algorithm_name, neighbors, community_neighbors, factor, prior_partner_factor) 
    
    def init_algorithm(name):
        algorithms = {
            'weighted_factor': init_weighted_factor,
            'weighted_factor_memory': init_weighted_factor_memory
        }
        return algorithms.get(name, init_default_algorithm)

    stop_event = threading.Event()
    service = GossipService(name, neighbors, init_algorithm(algorithm_name), nodeValue, stop_event)
    server = grpc.server(futures.ThreadPoolExecutor())
    gossip_pb2_grpc.add_GossipServicer_to_server(service, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    # Wait for the stop event to be set when stop application grpc call is invoked
    stop_event.wait()
    print("Stopping server...")
    server.stop(0)
    print("Server stopped.")
    print("Stopping application.")
    sys.exit(0)
