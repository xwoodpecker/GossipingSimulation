import random
from collections import defaultdict

import networkx as nx
from io import StringIO

# Sample adjacency list string (replace with your actual adjacency list)
from community import community_louvain

adjacency_list_string = """
0 2 3 4 6 7 16 17 19 23 25 27 33 46,
1 2,
2 5 7 19 26,
3 4 6 8 9 10 20 22 32 35 36,
4 5 12 16 21 24 29 31 38,
5 9 14 15 18 21 22 37,
6 10 20 29 34 44,
7 23,
8,
9 11 15 17 39,
10 12 14 30 38,
11,
12 13 24,
13 40 41,
14 36,
15 48,
16 44,
17 33 39 45,
18,
19,
20 28 46 47,
21,
22 28 35,
23 43,
24,
25 27,
26,
27,
28 47,
29 37,
30,
31,
32 34 49,
33,
34 49,
35,
36 43 45,
37,
38 42,
39,
40,
41,
42,
43,
44,
45,
46,
47,
48,
49,
"""


adjacency_list_string = """
0 1 4 5 8 10 20 39 42,
1 2 3 12 13 17 24 35,
2 15 23,
3 19 28 40,
4 9 14 30,
5 6 7 18 21 29 37 43,
6,
7 11,
8 26 36 44 45,
9 32,
10,
11,
12,
13 27,
14 16,
15,
16,
17,
18,
19 22 25 33 47,
20,
21 31 38,
22,
23,
24,
25 41,
26,
27,
28,
29 34,
30 46,
31,
32,
33,
34,
35,
36 48,
37,
38,
39,
40,
41,
42,
43 49,
44,
45,
46,
47,
48,
49,
"""

# Parse the adjacency list string and create the graph
G = nx.Graph()
for line in adjacency_list_string.strip().split('\n'):
    nodes = line.strip().split(',')
    source, *neighbors = map(int, nodes[0].split())
    G.add_node(source)
    G.add_edges_from((source, neighbor) for neighbor in neighbors)

# Compute hub and authority scores
hub_scores, authority_scores = nx.hits(G)

# Convert scores to arrays
hub_scores_array = [hub_scores[node] for node in sorted(G.nodes())]
authority_scores_array = [authority_scores[node] for node in sorted(G.nodes())]

print("Hub Scores:", hub_scores_array)
print("Authority Scores:", authority_scores_array)


# Compute degree centrality
degree_centralities = nx.degree_centrality(G)
print("degree_centralities:", degree_centralities)
# Compute degree centrality
betweenness_centralities = nx.betweenness_centrality(G)
print("betweenness_centralities:", betweenness_centralities)


partition = community_louvain.best_partition(G)
partition = {int(k): int(v) for k, v in partition.items()}

def get_community_node_dict(partition):
    # create a dictionary with node ids as keys and community ids as values
    # this is effectively a non-shallow copy of partition
    node_community_dict = {int(node): int(community_id) for node, community_id in partition.items()}
    # this dict contains the community ids as keys and the node ids as values
    community_node_dict = {}
    for node, community_id in node_community_dict.items():
        if community_id not in community_node_dict:
            community_node_dict[community_id] = [node]
        else:
            community_node_dict[community_id].append(node)
    return node_community_dict, community_node_dict

node_community_dict, community_node_dict = get_community_node_dict(partition)


# create a dictionary containing the nodes as keys
# and their respective neighbors as values
neighbors = {}
for node in G.nodes:
    neighbors[node] = list(G.neighbors(node))


# Compute the community_probabilities for each cluster for each node
community_probabilities = {}
for node, cluster in partition.items():
    if node not in community_probabilities:
        community_probabilities[node] = {}
    neighbor_count = len(list(neighbors[node]))
    for neighbor in neighbors[node]:
        neighbor_cluster = partition[neighbor]
        if neighbor_cluster not in community_probabilities[node]:
            community_probabilities[node][neighbor_cluster] = 0
        community_probabilities[node][neighbor_cluster] += 1 / neighbor_count

node_same_community_probabilities_neighbors = {}
for node in G.nodes:
    same_community_probabilities_neighbors = []
    for neighbor in neighbors[node]:
        neighbor_community_probabilities = community_probabilities[neighbor]
        same_community_probabilities_neighbors.append(neighbor_community_probabilities[node_community_dict[node]])
    node_same_community_probabilities_neighbors[node] = same_community_probabilities_neighbors


# Example usage
class Node:
    def __init__(self, number, neighbors, community):
        self.number = number
        self.neighbors = neighbors
        self.community = community

    def assign_community_probabilities(self, same_community_probabilities):
        self.same_community_probabilities = same_community_probabilities
        inverted_probabilities = [1 / x for x in self.same_community_probabilities]
        # set weights to inverted probabilities
        self.weights = {}
        for i in range(0, len(self.neighbors)):
            self.weights[self.neighbors[i]] = inverted_probabilities[i]

    def assign_neighboring_communities(self, neighboring_communities):
        community_members = defaultdict(set)
        for key, value in neighboring_communities.items():
            community_members[value].add(key)
        comm_count = len(community_members.keys())
        self.weights = {}
        for i in range(0, len(self.neighbors)):
            community_members_count = len(community_members[neighboring_communities[self.neighbors[i]]])
            self.weights[self.neighbors[i]] = (1/comm_count) * (1/community_members_count)

    def return_weights(self):
        return self.weights

    def simulate_select_neighbor(self):
        selected = random.choices(self.neighbors, weights=self.weights.values())[0]
        return selected

nodes = []
for n in G.nodes:
    node = Node(n, neighbors[n], node_community_dict[n])
    nodes.append(node)

for node in nodes:
    neighboring_communities = {}
    for neighbor in neighbors[node.number]:
        neighbor_community = node_community_dict[neighbor]
        neighboring_communities[neighbor] = neighbor_community
    print('Node: ', node.number)
    print('neighboring  weights:')
    node.assign_neighboring_communities(neighboring_communities)
    print(node.return_weights())
    print('community_probabilities  weights:')
    node.assign_community_probabilities(node_same_community_probabilities_neighbors[node.number])
    print(node.return_weights())

#for node in nodes:

#for node in nodes:
#    print('Node: ', node.number)
#    print(node.return_weights())
#    for i in range(0,5):
#        print(f'Simulation: {i}', node.simulate_select_neighbor())