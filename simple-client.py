import socket
import random
import os


class Client:
    def __init__(self):
        self.value = random.randint(0, 100)
        self.value_history = [(self.value, 0)]
        self.num_gossip_calls = 0
        self.neighbors = os.environ.get("neighbors").split(",")
        self.observer = os.environ.get("observer")

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as observer_sock:
            observer_sock.connect(self.observer)
            self.gossip()
            observer_sock.close()
            self.send_value(self.value_history)

    def send_value(self, sock):
        sock.sendall(bytes(str(self.value), 'utf-8'))

    def gossip(self):
        self.num_gossip_calls += 1
        peer_address = random.choice(self.neighbors)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as peer_sock:
            peer_sock.connect(tuple(peer_address.split(":")))
            peer_sock.sendall(bytes(str(self.value), 'utf-8'))
            data = peer_sock.recv(1024)
            peer_value = int(data.decode('utf-8'))
            self.value = min(self.value, peer_value)
            self.value_history.append((self.value, self.num_gossip_calls))

if __name__ == '__main__':
    client = Client()
    client.run()