adj_li = "1 2 4,2 3,3 4,4"
adj_li = adj_li.rstrip(',')

adjacency_list = []
for edge_str in adj_li.split(','):
    if edge_str:
        edge = tuple(map(int, edge_str.strip().split()))
        adjacency_list.append(edge)


entries = [split_str for split_str in adj_li.split(',')]

nodes = [entry[0] for entry in entries]

edge_dict = {}
for entry in entries:
    sub_entries = entry.split()
    key = sub_entries[0]
    edges = []
    for sub_entry in sub_entries[1:]:
        edge = (key, sub_entry)
        edges.append(edge)
        edge_dict[key] = edges

print(edge_dict)



nodes = [s[0] for s in adj_li.split(',')]
#print(nodes)


#for node in nodes:
    # Find the edges that connect to this node
    
    #print(f'Node {node} has edges: {edges}')

edges = []
for adj in adjacency_list:
    origin = adj[0]

    es =  ([(origin, neighbor) for neighbor in adj[1:]])
    if es:
        edges = edges + es

#print(edges)

name="my-graph"
pod_dict = {}
# Create a Pod for each node in the graph
for node in nodes:
    # Create a Pod for this node
    pod_name = f'{name}-node-{node}'
    pod_dict[node] = pod_name

#print(pod_dict)

for edge in edges:
    node1 = pod_dict[str(edge[0])]
    node2 = pod_dict[str(edge[1])]

    if node1 and node2:
        service_name = f'{name}-service-{node1}-{node2}'

        labels = {
            'app': 'gossip',
            'graph': name,
            'node': str(edge[0])
        }

        selector = {
            'app': 'gossip',
            'graph': name,
            'node': str(edge[1])
        }

        #print(edge)
        #print(labels)
        #print(selector)