import random
import community
import networkx as nx

adj_list = "1 2 3,2 3,3 4,4 5,5 6 7,6 7,7 8,8"
graph = nx.parse_adjlist(adj_list.split(','))

# Compute the best partition
partition = community.best_partition(graph)

# Compute the cluster sizes
cluster_sizes = {}
for node, cluster in partition.items():
    if cluster not in cluster_sizes:
        cluster_sizes[cluster] = 0
    cluster_sizes[cluster] += 1

# Compute the probabilities for each cluster for each node
probabilities = {}
for node, cluster in partition.items():
    if node not in probabilities:
        probabilities[node] = {}
    neighbors = graph.neighbors(node)
    neighbor_count = len(list(graph.neighbors(node)))
    for neighbor in neighbors:
        neighbor_cluster = partition[neighbor]
        if neighbor_cluster not in probabilities[node]:
            probabilities[node][neighbor_cluster] = 0
        probabilities[node][neighbor_cluster] += 1 / neighbor_count

# Print the probabilities for each cluster for each node
for node, cluster_probs in probabilities.items():
    print(f"Node {node}:")
    for cluster, prob in cluster_probs.items():
        print(f"Cluster {cluster}: {prob}") 
    #print(f'Louvain Cluster: {partition[node]}')

cluster_probs = probabilities["5"]
print(cluster_probs)
test = cluster_probs[0]
print(test)