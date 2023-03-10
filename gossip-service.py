import grpc
import random
import time
import gossip_pb2
import gossip_pb2_grpc
import socket
import os
from concurrent import futures


class GossipService(gossip_pb2_grpc.GossipServicer):
    def __init__(self):
        self.value = random.randint(0, 100)
        self.value_history = [(0, self.value)]
        self.num_gossip_calls = 0
        print('GossipService initialized with value {}'.format(self.value))
        #self.neighbors = os.environ.get("neighbors").split(",")
        #self.observer = os.environ.get("observer")


    def gossip(self):
        print('Starting to gossip.')
        self.num_gossip_calls += 1
        self.value -= 1
        self.value_history.append((self.num_gossip_calls, self.value))
        #peer_address = random.choice(self.neighbors)
        #with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as peer_sock:
        #    peer_sock.connect(tuple(peer_address.split(":")))
        #    peer_sock.sendall(bytes(str(self.value), 'utf-8'))
        #    data = peer_sock.recv(1024)
        #    peer_value = int(data.decode('utf-8'))
        #    self.value = min(self.value, peer_value)
        #    self.value_history.append(self.num_gossip_calls, self.value))
        print('Finished gossiping.')


    def Gossip(self, request, context):
        self.gossip()
        
    def Gossip(self, request, context):
        print('GRPC Gossip invoked.')
        self.gossip()
        last_entry = self.value_history[-1]
        return gossip_pb2.GossipResponse(
            value_entry=gossip_pb2.ValueEntry(num_gossip_calls=last_entry[0], value=last_entry[1]))
    
    def History(self, request, context):
        print('GRPC History invoked.')
        return gossip_pb2.HistoryResponse(
            value_history=[gossip_pb2.ValueEntry(value=v, num_gossip_calls=c) for v, c in self.value_history])
    
def serve():
    server = grpc.server(futures.ThreadPoolExecutor())
    gossip_pb2_grpc.add_GossipServicer_to_server(GossipService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()