import time
import json
import kopf
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import networkx as nx
import community as community_louvain
from cfg import *

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
    split_adj_list = str_adj_list.split(',')
    adjacency_list = []
    for edge_str in split_adj_list:
        if edge_str:
            edge = tuple(map(int, edge_str.strip().split()))
            adjacency_list.append(edge)

    logger.info(f'The graph has the adjacency list: {adjacency_list}')

    entries = [split_str for split_str in split_adj_list]
    nodes = [entry[0] for entry in entries]
    nodes.sort()

    logger.info(f'The graph has the nodes: {nodes}')

    neighbors = {}
    for node in nodes:
        neighbors[node] = []

    # Construct neighbors for each node
    for entry in entries:
        sub_entries = entry.split()
        key = sub_entries[0]
        for sub_entry in sub_entries[1:]:
            neighbors[key].append(sub_entry)
            neighbors[sub_entry].append(key)
    neighbors = {key: sorted(values) for key, values in sorted(neighbors.items())}

    logger.info(f'Neighbors of each node: {neighbors}')

    algorithm = spec.get('algorithm', DEFAULT_ALGORITHM)
    logger.info(f'Simulation running algorithm {algorithm}')
    
    def get_community_node_dict(partition):
        # create a dictionary with node ids as keys and community ids as values
        node_community_dict = {node: community_id for node, community_id in partition.items()}
        community_node_dict = {}
        for node, community_id in partition.items():
            if community_id not in community_node_dict:
                community_node_dict[community_id] = [node]
            else:
                community_node_dict[community_id].append(node)   
        logger.info(f'Node communities: {node_community_dict}')
        return node_community_dict, community_node_dict

    # comminities are needed for weighted_factor and community probability assignment
    if algorithm in WEIGHTED_FACTOR_SET + COMMUNITY_PROBABILITIES_SET:
        graph = nx.parse_adjlist(split_adj_list)
        # apply louvain method on the graph
        partition = community_louvain.best_partition(graph)
        node_community_dict, community_node_dict = get_community_node_dict(partition)

        if algorithm in WEIGHTED_FACTOR_SET:
            factor = spec.get('factor', DEFAULT_FACTOR)

        if algorithm in COMMUNITY_PROBABILITIES_SET:
            # Compute the cluster sizes
            cluster_sizes = {}
            for node, cluster in partition.items():
                if cluster not in cluster_sizes:
                    cluster_sizes[cluster] = 0
                cluster_sizes[cluster] += 1

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

        if algorithm in MEMORY_SET:
            prior_partner_factor = spec.get('priorPartnerFactor', DEFAULT_ALGORITHM)
    
    randomInitialization = spec.get('randomInitialization', True)
    if not randomInitialization:
        str_value_list = graph_spec.get('valueList', '').rstrip(',')
        split_value_list = str_value_list.split(',')
        if len(split_value_list) == len(nodes):
            values = split_value_list
        else: 
            # set the node value to the node number
            values = nodes
        node_values = {}
        for i in range(len(nodes)):
            node_values[nodes[i]] = values[i]

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
            
            env = []

            neighbors_str = ','.join([f'{name}-{graph_name}-node-{n}' for n in neighbors[node]])
            env.append(client.V1EnvVar(name=ENVIRONMENT_NEIGHBORS, value=neighbors_str))
            
            env.append(client.V1EnvVar(name=ENVIRONMENT_ALGORITHM, value=algorithm))

            env.append(client.V1EnvVar(name=ENVIRONMENT_RANDOM_INITIALIZATION, value=str(randomInitialization)))
            if not randomInitialization:
                env.append(client.V1EnvVar(name=ENVIRONMENT_NODE_VALUE, value=str(node_values[node])))

            if algorithm in WEIGHTED_FACTOR_SET:
                community_id = node_community_dict[node]
                community_nodes = community_node_dict[community_id]
                # extract community and non-community neighbors
                neighbor_nodes = set(neighbors[node])
                community_neighbors = []
                #non_community_neighbors = []
                for neighbor_node in neighbor_nodes:
                    if neighbor_node in community_nodes:
                        community_neighbors.append(neighbor_node)
                    #else:
                    #    non_community_neighbors.append(neighbor_node)

                community_neighbors_str = ','.join([f'{name}-{graph_name}-node-{n}' for n in community_neighbors])
                env.append(client.V1EnvVar(name=ENVIRONMENT_COMMUNITY_NEIGHBORS, value=community_neighbors_str))
                #non_community_neighbors_str = ','.join([f'{name}-{graph_name}-node-{n}' for n in non_community_neighbors])
                #env.append(client.V1EnvVar(name='NON_COMMUNITY_NEIGHBORS', value=non_community_neighbors_str))
                env.append(client.V1EnvVar(name=ENVIRONMENT_FACTOR, value=str(factor)))

            if algorithm in COMMUNITY_PROBABILITIES_SET:
                community_id = node_community_dict[node]
                neighbor_nodes = neighbors[node]

                same_community_probabilities_neighbors = []
                for neighbor in neighbor_nodes:
                    neighbor_community_probabilities = community_probabilities[neighbor]
                    same_community_probabilities_neighbors.append(neighbor_community_probabilities[community_id])
                
                same_community_probabilities_neighbors_str = ','.join(
                    str(round(item, COMMUNITY_PROBABILITIES_ROUNDING)) 
                    for item 
                    in same_community_probabilities_neighbors
                )
                env.append(client.V1EnvVar(name=ENVIRONMENT_SAME_COMMUNITY_PROBABILITIES_NEIGHBORS, value=same_community_probabilities_neighbors_str))

            if algorithm in MEMORY_SET:
                env.append(client.V1EnvVar(name=ENVIRONMENT_PRIOR_PARTNER_FACTOR, value=str(prior_partner_factor)))

             # Create the container for the Pod
            container = client.V1Container(
            name=DOCKER_NODE_NAME,
            image=DOCKER_NODE_IMAGE,
            env=env,
            ports=[
                client.V1ContainerPort(container_port=TCP_SERVICE_PORT, name='tcp'),
                client.V1ContainerPort(container_port=GRPC_SERVICE_PORT, name='grpc')
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
                port=TCP_SERVICE_PORT,
                target_port='tcp'
            ),
            client.V1ServicePort(
                name='grpc',
                port=GRPC_SERVICE_PORT,
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

    repetitions = spec.get('repetitions', 1)
    visualize = spec.get('visualize', False)
    simulationProperties = spec.get('simulationProperties', {})

    graphType = graph_spec.get('graphType', 'normal')
    modularity = graph_spec.get('modularity', None)
    graphProperties = graph_spec.get('graphProperties', {})

    def create_runner_pod():

        pod_name = f'{name}-runner'

        labels = {
                    'app': 'gossip',
                    'simulation': name,
                    'graph': graph_name,
                    'node': 'runner'
                }
        
        nodes_str = ','.join(pods)
        
        env = []

        env.append(client.V1EnvVar(name=ENVIRONMENT_SIMULATION, value=name))
        env.append(client.V1EnvVar(name=ENVIRONMENT_ALGORITHM, value=algorithm))
        env.append(client.V1EnvVar(name=ENVIRONMENT_REPETITIONS, value=str(repetitions)))
        env.append(client.V1EnvVar(name=ENVIRONMENT_ADJ_LIST, value=str_adj_list))
        env.append(client.V1EnvVar(name=ENVIRONMENT_NODES, value=nodes_str))

        # set for graph highlighting in plots
        if algorithm in NODE_COMMUNITIES_SET:
            node_community_string = json.dumps(node_community_dict)
            env.append(client.V1EnvVar(name=ENVIRONMENT_NODE_COMMUNITIES, value=node_community_string))

        env.append(client.V1EnvVar(name=ENVIRONMENT_VISUALIZE, value=str(visualize)))

        simulation_properties = simulationProperties.copy()
        simulation_properties_string = json.dumps(simulation_properties)
        env.append(client.V1EnvVar(name=ENVIRONMENT_SIMULATION_PROPERTIES, value=simulation_properties_string))

        graph_properties = graphProperties.copy()
        graph_properties['GraphType'] = graphType
        if modularity:
            graph_properties['Modularity'] = modularity
        graph_properties_string = json.dumps(graph_properties)
        env.append(client.V1EnvVar(name=ENVIRONMENT_GRAPH_PROPERTIES, value=graph_properties_string))

        # Create the container for the Pod
        container = client.V1Container(
        name=DOCKER_RUNNER_NAME,
        image=DOCKER_RUNNER_IMAGE,
        env=env,
        env_from=[
            client.V1EnvFromSource(
                config_map_ref=client.V1ConfigMapEnvSource(name=MINIO_CONFIGMAP_NAME)
            ),
            client.V1EnvFromSource(
                secret_ref=client.V1SecretEnvSource(name=MINIO_SECRETS_NAME)
            )
        ],
        ports=[
            client.V1ContainerPort(container_port=GRPC_SERVICE_PORT, name='grpc')
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
            port=GRPC_SERVICE_PORT,
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
    create_pods()
    
    labels = {
            'app': 'gossip',
            'simulation': name,
            'graph': graph_name
        }

    while True:
        # List all pods matching the specified labels
        pods = api.list_namespaced_pod(namespace=namespace,
                                        label_selector=','.join([f"{k}={v}" for k, v in labels.items()]))

        # Check if all pods are in the "Running" state
        if all(pod.status.phase == 'Running' for pod in pods.items):
            break  # All pods are running, exit the loop
        time.sleep(1)  # Wait for 1 second before checking again

    # All pods are running, create the runner pod
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