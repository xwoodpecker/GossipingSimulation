import kopf
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import time

# Load in-cluster configuration
config.load_incluster_config()


@kopf.on.create('gossip.io', 'v1', 'graphs')
def create_pods_for_graph(spec, name, namespace, logger, **kwargs):
    api = client.CoreV1Api()

    # Convert the adjacency list from a comma-separated string to a list of tuples
    str_adj_list = spec.get('adjacencyList', '').rstrip(',')
    adjacency_list = []
    for edge_str in str_adj_list.split(','):
        if edge_str:
            edge = tuple(map(int, edge_str.strip().split()))
            adjacency_list.append(edge)
    logger.info(f'The graph has the adjacency list: {adjacency_list}')

    nodes = [split_str[0] for split_str in str_adj_list.split(',')]
    logger.info(f'The graph has the nodes: {nodes}')

    edges = []
    for adj in adjacency_list:
        origin = adj[0]
        es = ([(origin, neighbor) for neighbor in adj[1:]])
        if es:
            edges = edges + es
    logger.info(f'The graph has the edges: {edges}')

    def create_pods():

        # Create a dictionary to store the Pod names and their node number
        pod_dictionary = {}
        # Create a Pod for each node in the graph
        for node in nodes:
            # Create a Pod for this node
            pod_name = f'{name}-node-{node}'
            pod_dictionary[node] = pod_name

            pod = client.V1Pod(
                metadata=client.V1ObjectMeta(
                    name=pod_name,
                    namespace=namespace,
                    labels={
                        'app': 'gossip',
                        'graph': name,
                        'node': str(node)
                    }
                ),
                spec=client.V1PodSpec(
                    containers=[
                        client.V1Container(
                            name='node-container',
                            image='ubuntu',
                            command=['bash', '-c', f'echo "This is node {node}" && sleep 3600']
                        )
                    ]
                )
            )
            try:
                api.create_namespaced_pod(namespace=namespace, body=pod)
                logger.info(f'Pod {pod_name} created')
            except ApiException as e:
                logger.error(f'Error creating pod: {e}')

        logger.info(f'Finished creating Pods for graph {name}')

        return pod_dictionary

    def create_services():

        for e in edges:
            node1 = pod_dict[str(e[0])]
            node2 = pod_dict[str(e[1])]

            # naming convention always node with smaller index first (?)
            # export naming function to a common utils file

            if node1 and node2:
                service_name = f'{name}-service-{node1}-{node2}'

                # this does not work I think
                labels = {
                    'app': 'gossip',
                    'graph': name
                }
                selector = {
                    'app': 'gossip',
                    'graph': name,
                    'matchExpressions': [
                        {
                            'key': 'node',
                            'operator': 'In',
                            'values': [str(e[0]), str(e[1])]
                        }
                    ]
                }

                port = client.V1ServicePort(
                    name='tcp',
                    port=80,
                    target_port=80
                )

                service = client.V1Service(
                    metadata=client.V1ObjectMeta(
                        name=service_name,
                        labels=labels
                    ),
                    spec=client.V1ServiceSpec(
                        selector=selector,
                        ports=[port],
                        type='ClusterIP'
                    )
                )

                try:
                    api.create_namespaced_service(namespace=namespace, body=service)
                    logger.info(f'Service {service_name} created')
                except ApiException as e:
                    logger.error(f'Error creating service: {e}')

        logger.info(f'Finished creating Services for graph {name}')

    pod_dict = create_pods()

    create_services()

    logger.info(f'Finished creating Graph {name}')
