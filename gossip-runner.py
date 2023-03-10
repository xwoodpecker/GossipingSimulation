import grpc
import gossip_pb2
import gossip_pb2_grpc
import time


class GossipRunner:
    def __init__(self, endpoints):
        self.endpoints = endpoints
        self.stubs = [gossip_pb2_grpc.GossipStub(grpc.insecure_channel(endpoint)) for endpoint in endpoints]

    def run(self):
        round_num = 1
        while True:
            print(f"Starting round {round_num} of gossiping...")
            value_entries = []
            for stub in self.stubs:
                response = stub.Gossip(gossip_pb2.GossipRequest())
                print(f"Received value_entry : {response.value_entry}")
                value_entries.append(response.value_entry)
            #last_rounds = [value_entry[-1].num_gossip_calls for value_entry in value_entries]
            #if all(key == last_rounds[0] for key in last_rounds):
            #    values = [value_entry[-1].value for value_entry in value_entries]
            #    if all(value == values[0] for value in values):
            #        print(f"All hosts have converged on value {values[0]}")
            #        break
            round_num += 1
            time.sleep(2)


if __name__ == '__main__':
    runner = GossipRunner(['localhost:50051'])
    runner.run()