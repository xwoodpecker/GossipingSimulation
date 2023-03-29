import kopf
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import time

# Load in-cluster configuration
config.load_incluster_config()
api = client.CoreV1Api()
customs_api = client.CustomObjectsApi()

@kopf.on.create('gossip.io', 'v1', 'simulations')
def create_pods_for_simulation(spec, name, namespace, logger, **kwargs):

    graph_selector = spec.get('graphSelector', {})
    match_labels = graph_selector.get('matchLabels', {})

    graph_name = match_labels['app']
    graph_obj = customs_api.get_namespaced_custom_object('gossip.io', 'v1', namespace, 'graphs', graph_name)
    graph_spec = graph_obj['spec']

    logger.info(f'Selected graph {graph_name} from graphSelector.')

    # Convert the adjacency list from a comma-separated string to a list of tuples
    str_adj_list = graph_spec.get('adjacencyList', '').rstrip(',')
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

    pods = []

    def create_pods():

        # Create a dictionary to store the Pod names and their node number
        # Create a Pod for each node in the graph
        for node in nodes:
            # Create a Pod for this node
            pod_name = f'{name}-{graph_name}-node-{node}'

            labels = {
                        'app': 'gossip',
                        'simulation': name,
                        'graph': graph_name,
                        'node': str(node)
                    }
            
            neighbors_str = ','.join([f'{name}-{graph_name}-node-{n}' for n in neighbors[node]])
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
                logger.info(f'Pod {pod_name} created.')
                pods.append(pod_name)
            except ApiException as e:
                logger.error(f'Error creating pod: {e}')

        logger.info(f'Finished creating Pods for simulation {name} on graph {graph_name}.')

   
    
    def create_services():

        for node in nodes:

            service_name = f'{name}-{graph_name}-node-{node}'

            # this does not work I think
            labels = {
                'app': 'gossip',
                'simulation': name,
                'graph': graph_name,
                'node': str(node)
            }
            selector = {
                'app': 'gossip',
                'simulation': name,
                'graph': graph_name,
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
                logger.info(f'Service {service_name} created.')
            except ApiException as e:
                logger.error(f'Error creating service: {e}')

        logger.info(f'Finished creating Services for simulation {name} on graph {graph_name}.')


    def create_runner_pod():

        pod_name = f'{name}-runner'

        labels = {
                    'app': 'gossip',
                    'simulation': name,
                    'graph': graph_name,
                    'node': 'runner'
                }
        
        nodes_str = ','.join(pods)
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

        logger.info(f'Finished creating simulation runner pod for simulation {name}.')


    def create_runner_service():

        service_name = f'{name}-runner'

        labels = {
            'app': 'gossip',
            'simulation': name,
            'graph': graph_name,
            'node': 'runner'
        }
        selector = {
            'app': 'gossip',
            'simulation': name,
            'graph': graph_name,
            'node': 'runner'
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

        logger.info(f'Finished creating simulation runner service for simulation {name}.')

    create_services()
    create_runner_service()
    time.sleep(3)
    create_pods()
    time.sleep(3)
    create_runner_pod()
    logger.info(f'Finished creating resources for simulation {name}.')

@kopf.on.delete('gossip.io', 'v1', 'simulations')
def delete_services_and_pods(body, **kwargs):
    simulation_name = body['metadata']['name']
    namespace = body['metadata']['namespace']
    
    # Delete all services with the simulation name label
    services = api.list_namespaced_service(namespace, label_selector=f"simulation={simulation_name}")
    for service in services.items:
        api.delete_namespaced_service(service.metadata.name, namespace)
        
    # Delete all pods with the simulation name label
    pods = api.list_namespaced_pod(namespace, label_selector=f"simulation={simulation_name}")
    for pod in pods.items:
        api.delete_namespaced_pod(pod.metadata.name, namespace)