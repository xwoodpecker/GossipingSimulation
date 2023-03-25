import grpc
import random
import time
import sys
import socket
import os
from concurrent import futures
import threading

sys.path.append("/app/grpc_compiled")
import gossip_pb2
import gossip_pb2_grpc


class ValueEntry:
    def __init__(self, participations, value):
        self.participations = participations
        self.value = value     

class GossipService(gossip_pb2_grpc.GossipServicer):
    
    def __init__(self):
        self.value = random.randint(0, 100)
        self.participations = 0
        self.value_entries = []
        self.value_entries.append(ValueEntry(0, self.value))
        self.neighbors = os.environ.get("NEIGHBORS").rstrip(',').split(",")
        self.name = os.environ.get("HOSTNAME")
        print(f'GossipService initialized on {self.name} with value {self.value}.')
        print(f"Received neighbors: {self.neighbors}")

    def process_gossip_data(self, data):
        self.participations += 1
        peer_value = int(data.decode('utf-8'))
        print(f"Received peer value: {peer_value}, Current value: {self.value}")
        self.value = min(self.value, peer_value)
        print(f"Value after gossiping: {self.value}")
        self.value_entries.append(ValueEntry(self.participations, self.value))


    def gossip(self):
        print('Starting to gossip.')
        neighbor = random.choice(self.neighbors)
        print(f"Selected neighbor {neighbor} for gossiping.")
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

    def listen_for_connections(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.bind((self.name, 90))
                server_socket.listen(1)
                print('Listening on port 90 for incoming gossip...')
                while True:
                    conn, addr = server_socket.accept()
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

def serve():
    service = GossipService()
    server = grpc.server(futures.ThreadPoolExecutor())
    gossip_pb2_grpc.add_GossipServicer_to_server(service, server)
    server.add_insecure_port('[::]:50051')
    server.start()
      # Start the listen_for_connections method in a background thread
    listen_thread = threading.Thread(target=service.listen_for_connections)
    listen_thread.start()
    print("Server started on port 50051")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
