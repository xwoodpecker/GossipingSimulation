import grpc
import gossip_pb2
import gossip_pb2_grpc
import time
import socket


class GossipRunner:
    def __init__(self, ip_addresses):
        self.ip_addresses = ip_addresses
        self.stubs = [gossip_pb2_grpc.GossipStub(grpc.insecure_channel(f"{ip}:50051")) for ip in ip_addresses]

    def run(self):
        round_num = 1
        while True:
            print(f"Starting round {round_num} of gossiping...")
            value_entries = []
            for stub in self.stubs:
                response = stub.Gossip(gossip_pb2.GossipRequest())
                print(f"Received value_entry : {response.value_entry}")
                value_entries.append(response.value_entry)
            last_rounds = [value_entry[-1].num_gossip_calls for value_entry in value_entries]
            if all(key == last_rounds[0] for key in last_rounds):
                values = [value_entry[-1].value for value_entry in value_entries]
                if all(value == values[0] for value in values):
                    print(f"All hosts have converged on value {values[0]}")
                    break
            round_num += 1
            time.sleep(2)


if __name__ == '__main__':

    pod_ips = [socket.gethostbyname(name) for name in pod_names]
    runner = GossipRunner(pod_ips)
    runner.run()