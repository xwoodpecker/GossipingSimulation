import kopf
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import time

# Load in-cluster configuration
config.load_incluster_config()
api = client.CoreV1Api()

@kopf.on.create('gossip.io', 'v1', 'graphs')
def create_pods_for_graph(spec, name, namespace, logger, **kwargs):

    # Convert the adjacency list from a comma-separated string to a list of tuples
    str_adj_list = spec.get('adjacencyList', '').rstrip(',')
    adjacency_list = []
    for edge_str in str_adj_list.split(','):
        if edge_str:
            edge = tuple(map(int, edge_str.strip().split()))
            adjacency_list.append(edge)
    logger.info(f'The graph has the adjacency list: {adjacency_list}')


    entries = [split_str for split_str in str_adj_list.split(',')]

    nodes = [entry[0] for entry in entries]

    logger.info(f'The graph has the nodes: {nodes}')

    neighbors = {}
    for node in nodes:
        neighbors[node] = []

    for entry in entries:
        sub_entries = entry.split()
        key = sub_entries[0]
        
        for sub_entry in sub_entries[1:]:
            neighbors[key].append(sub_entry)
            neighbors[sub_entry].append(key)

    logger.info(f'Neighbors of each node: {neighbors}')

    def create_pods():

        # Create a dictionary to store the Pod names and their node number
        pod_dictionary = {}
        # Create a Pod for each node in the graph
        for node in nodes:
            # Create a Pod for this node
            pod_name = f'{name}-node-{node}'
            pod_dictionary[node] = pod_name

            labels = {
                        'app': 'gossip',
                        'graph': name,
                        'node': str(node)
                    }
            
            neighbors_str = ','.join([f'{name}-node-{n}' for n in neighbors[node]])
            env_var = client.V1EnvVar(name='NEIGHBORS', value=neighbors_str)

             # Create the container for the Pod
            container = client.V1Container(
            name='node-example',
            image='xwoodpecker/node-example:latest',
            env=[env_var],
            ports=[
                client.V1ContainerPort(container_port=90, name='tcp'),
                client.V1ContainerPort(container_port=50051, name='grpc')
            ]
        )

            pod = client.V1Pod(
                metadata=client.V1ObjectMeta(
                    name=pod_name,
                    namespace=namespace,
                    labels=labels
                ),
                 spec=client.V1PodSpec(
                    restart_policy='OnFailure',
                    containers=[container]
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

        for node in nodes:

            service_name = f'{name}-node-{node}'

            # this does not work I think
            labels = {
                'app': 'gossip',
                'graph': name,
                'node': str(node)
            }
            selector = {
                'app': 'gossip',
                'graph': name,
                'node': str(node)
            }
            ports = [
            client.V1ServicePort(
                name='tcp',
                port=90,
                target_port='tcp'
            ),
            client.V1ServicePort(
                name='grpc',
                port=50051,
                target_port='grpc'
            )
        ]

            service = client.V1Service(
                metadata=client.V1ObjectMeta(
                    name=service_name,
                    labels=labels
                ),
                spec=client.V1ServiceSpec(
                    selector=selector,
                    ports=ports,
                    cluster_ip=None
                )
            )

            try:
                api.create_namespaced_service(namespace=namespace, body=service)
                logger.info(f'Service {service_name} created')
            except ApiException as e:
                logger.error(f'Error creating service: {e}')

        logger.info(f'Finished creating Services for graph {name}')


    def create_runner_pod(pod_dictionary):

        pod_name = f'{name}-simulation-runner'

        labels = {
                    'app': 'gossip',
                    'graph': name,
                    'node': 'simulation-runner'
                }
        
        nodes_str = ','.join(pod_dictionary.values())
        env_var = client.V1EnvVar(name='NODES', value=nodes_str)

        # Create the container for the Pod
        container = client.V1Container(
        name='runner-example',
        image='xwoodpecker/runner-example:latest',
        env=[env_var],
        ports=[
            client.V1ContainerPort(container_port=50051, name='grpc')
        ]
    )

        pod = client.V1Pod(
            metadata=client.V1ObjectMeta(
                name=pod_name,
                namespace=namespace,
                labels=labels
            ),
                spec=client.V1PodSpec(
                restart_policy='OnFailure',
                containers=[container]
            )
        )
        try:
            api.create_namespaced_pod(namespace=namespace, body=pod)
            logger.info(f'Pod {pod_name} created')
        except ApiException as e:
            logger.error(f'Error creating pod: {e}')

        logger.info(f'Finished creating simulation runner pod for graph {name}')


    def create_runner_service():
        service_name = f'{name}-simulation-runner'

        # this does not work I think
        labels = {
            'app': 'gossip',
            'graph': name,
            'node': 'simulation-runner'
        }
        selector = {
            'app': 'gossip',
            'graph': name,
            'node': 'simulation-runner'
        }
        ports = [
        client.V1ServicePort(
            name='grpc',
            port=50051,
            target_port='grpc'
        )
    ]

        service = client.V1Service(
            metadata=client.V1ObjectMeta(
                name=service_name,
                labels=labels
            ),
            spec=client.V1ServiceSpec(
                selector=selector,
                ports=ports,
                cluster_ip=None
            )
        )

        try:
            api.create_namespaced_service(namespace=namespace, body=service)
            logger.info(f'Service {service_name} created')
        except ApiException as e:
            logger.error(f'Error creating service: {e}')

        logger.info(f'Finished creating simulation runner service for graph {name}')

    create_services()
    pod_dict = create_pods()
    create_runner_service()
    create_runner_pod(pod_dict)
    logger.info(f'Finished creating Graph {name}')




## Define a handler function for the cleanup
#@kopf.on.timer('gossip.io', 'v1', interval=30)
#def cleanup(namespace, **kwargs):
#
#    # Define a function to check if the pods have completed
#    def check_pods_completion(namespace, pod_dictionary):
#        completed_pods = []
#        for node, pod_name in pod_dictionary.items():
#            try:
#                pod = api.read_namespaced_pod(pod_name, namespace)
#                if pod.status.phase == 'Succeeded':
#                    completed_pods.append(pod_name)
#            except ApiException as e:
#                logger.error(f'Error checking pod status: {e}')
#
#        # If all pods have completed, perform cleanup
#        if len(completed_pods) == len(pod_dictionary):
#            for pod_name in pod_dictionary.values():
#                try:
#                    api.delete_namespaced_pod(pod_name, namespace)
#                    logger.info(f'Pod {pod_name} deleted')
#                except ApiException as e:
#                    logger.error(f'Error deleting pod: {e}')
#                    
#    # Check if the pods have completed and perform cleanup if necessary
#    check_pods_com'pletion(namespace, pod_dict)