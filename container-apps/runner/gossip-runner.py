import os
import grpc
import time
import socket
import  sys

sys.path.append("/app/grpc_compiled")
import gossip_pb2
import gossip_pb2_grpc


class GossipRunner:
    def __init__(self, nodes):
        self.nodes = nodes
        self.stubs = [gossip_pb2_grpc.GossipStub(grpc.insecure_channel(f"{node}:50051")) for node in nodes]
        self.stub_dict = {node: stub for node, stub in zip(nodes, self.stubs)}
        print(f"Stubs set to {self.stub_dict}")

    def init_value_history(self):
        self.value_history = {}
        for node in self.stub_dict: 
            response = self.stub_dict[node].CurrentValue(gossip_pb2.CurrentValueRequest())
            self.value_history[node] = [(0, response.value)]

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
                val = response.value
                values.append(val)
                print(f"Node {node} has value {val}.")
                self.value_history[node].append((round_num, val))

            if all(value == values[0] for value in values):
                print(f"All hosts have converged on value {values[0]}")
                break
            round_num += 1
            time.sleep(1)
        print(f"The full value history for this run: {self.value_history}")

    def stop_node_applications(self):
        print(f"Stopping node applications...")
        for node in self.stub_dict: 
            response = self.stub_dict[node].StopApplication(gossip_pb2.StopApplicationRequest())
            print(f"Sent stop application request to node {node}")



if __name__ == '__main__':

    nodes = os.environ.get("NODES").rstrip(',').split(",")
    print(f"Received nodes: {nodes}")
    runner = GossipRunner(nodes)
    runner.init_value_history()
    runner.run()
    runner.stop_node_applications()
    print("Stopping application.")
    sys.exit(0)