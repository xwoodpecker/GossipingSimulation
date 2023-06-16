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
def create_services_and_pods(spec, name, namespace, logger, **kwargs):
    """
    Create pods and services for a simulation resource object.

    Args:
        spec (dict): The specification of the simulation.
        name (str): The name of the simulation.
        namespace (str): The namespace in which the simulation is created.
        logger (logging.Logger): The logger for logging messages.
        **kwargs: Additional keyword arguments.

    Returns:
        None
    """
    graph_selector = spec.get('graphSelector', {})
    match_labels = graph_selector.get('matchLabels', {})

    # Select graph from graphSelector
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

    # get and sort all the nodes in the adj list
    entries = [split_str for split_str in split_adj_list]
    nodes = [int(entry.split()[0]) for entry in entries]
    nodes.sort()

    # create a dictionary containing the nodes as keys
    # and their respective neighbors as values
    neighbors = {}
    for node in nodes:
        neighbors[node] = []

    # Construct neighbors for each node
    for entry in entries:
        sub_entries = entry.split()
        key = int(sub_entries[0])
        for sub_entry in sub_entries[1:]:
            sub_entry = int(sub_entry)
            neighbors[key].append(sub_entry)
            neighbors[sub_entry].append(key)
    neighbors = {key: sorted(values) for key, values in sorted(neighbors.items())}

    logger.info(f'Neighbors of each node: {neighbors}')

    algorithm = spec.get('algorithm', DEFAULT_ALGORITHM)
    logger.info(f'Simulation running algorithm {algorithm}')

    def get_community_node_dict(partition):
        """
        Get dictionaries mapping community IDs to node IDs and vice versa.

        Args:
            partition (dict): A dictionary with node IDs as keys and community IDs as values.

        Returns:
            tuple: A tuple containing two dictionaries:
                - node_community_dict (dict): A dictionary mapping node IDs to community IDs.
                - community_node_dict (dict): A dictionary mapping community IDs to lists of node IDs.
        """
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
        logger.info(f'Node communities: {node_community_dict}')
        return node_community_dict, community_node_dict

    # communities are needed for weighted_factor and community probability assignment
    if algorithm in NODE_COMMUNITIES_SET:
        graph = nx.parse_adjlist(split_adj_list)
        # apply louvain method on the graph
        partition = community_louvain.best_partition(graph)
        node_community_dict, community_node_dict = get_community_node_dict(partition)

        # weighted factor algorithms use a factor to modify the probability 
        # of selecting a partner inside or outside the community
        if algorithm in WEIGHTED_FACTOR_SET:
            factor = spec.get('factor', DEFAULT_FACTOR)

        # community probability algorithms use statistical data for each selection
        # of the next gossip partner
        if algorithm in COMMUNITY_PROBABILITIES_SET:
            # Compute the cluster sizes
            cluster_sizes = {}
            for node, cluster in partition.items():
                if cluster not in cluster_sizes:
                    cluster_sizes[cluster] = 0
                cluster_sizes[cluster] += 1

            # Compute the community_probabilities for each cluster for each node
            # It represents the certainty of the node belonging to the cluster
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

    # memory algorithms use a factor to modify the probability 
    # of selecting a partner that has already been selected in a previous gossiping
    if algorithm in MEMORY_SET:
        prior_partner_factor = spec.get('priorPartnerFactor', DEFAULT_ALGORITHM)

    # random initialization sets the node value of each node
    randomInitialization = spec.get('randomInitialization', True)
    if not randomInitialization:
        str_value_list = graph_spec.get('valueList', '').rstrip(',')
        split_value_list = str_value_list.split(',')
        # set the values to the provided value list if the lengths match
        if len(split_value_list) == len(nodes):
            values = split_value_list
        else:
            # set the node value to the node number
            values = nodes
        # create a dict mapping nodes to their respective value
        node_values = {}
        for i in range(len(nodes)):
            node_values[nodes[i]] = values[i]

    pods = []

    def get_resource_name(simulation_name, node):
        """
        Generate a resource name based on the simulation name and node ID.

        Args:
            simulation_name (str): The name of the simulation.
            node (str): The node ID.

        Returns:
            str: The generated resource name.
        """
        return f'{name}-node-{node}'

    def create_node_pods():
        """
        Create node pods for the simulation.

        Returns:
            None
        """

        # Create a dictionary to store the Pod names and their node number
        # Create a Pod for each node in the graph
        for node in nodes:
            # Create a Pod for this node
            pod_name = get_resource_name(name, node)

            labels = {
                'app': 'gossip',
                'simulation': name,
                'graph': graph_name,
                'node': str(node)
            }

            env = []

            # set environment variables
            neighbors_str = ','.join([get_resource_name(name, n) for n in neighbors[node]])
            env.append(client.V1EnvVar(name=ENVIRONMENT_NEIGHBORS, value=neighbors_str))
            env.append(client.V1EnvVar(name=ENVIRONMENT_ALGORITHM, value=algorithm))
            env.append(client.V1EnvVar(name=ENVIRONMENT_RANDOM_INITIALIZATION, value=str(randomInitialization)))
            if not randomInitialization:
                env.append(client.V1EnvVar(name=ENVIRONMENT_NODE_VALUE, value=str(node_values[node])))

            # weighted factor algorithm specific environment variables
            if algorithm in WEIGHTED_FACTOR_SET:
                # set the community neighbors of the current node
                community_id = node_community_dict[node]
                community_nodes = community_node_dict[community_id]
                # extract community and non-community neighbors
                neighbor_nodes = set(neighbors[node])
                community_neighbors = []
                # non_community_neighbors = []
                for neighbor_node in neighbor_nodes:
                    if neighbor_node in community_nodes:
                        community_neighbors.append(neighbor_node)

                community_neighbors_str = ','.join([get_resource_name(name, n) for n in community_neighbors])
                env.append(client.V1EnvVar(name=ENVIRONMENT_COMMUNITY_NEIGHBORS, value=community_neighbors_str))
                env.append(client.V1EnvVar(name=ENVIRONMENT_FACTOR, value=str(factor)))

            # community probabilities algorithm specific environment variables
            if algorithm in COMMUNITY_PROBABILITIES_SET:
                # set the same community probabilities of the neighbors for the current node
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
                env.append(client.V1EnvVar(name=ENVIRONMENT_SAME_COMMUNITY_PROBABILITIES_NEIGHBORS,
                                           value=same_community_probabilities_neighbors_str))

            # memory algorithm specific environment variables
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

            # define the pod
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
                # create the pod in the current namespace
                api.create_namespaced_pod(namespace=namespace, body=pod)
                logger.info(f'Pod {pod_name} created.')
                pods.append(pod_name)
            except ApiException as e:
                logger.error(f'Error creating pod: {e}')

        logger.info(f'Finished creating Pods for simulation {name} on graph {graph_name}.')

    def create_node_services():
        """
        Create pods for the simulation.

        Returns:
            None
        """
        for node in nodes:
            # Create a Service for this node
            service_name = get_resource_name(name, node)

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

            # define the service
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
                # create the service in the current namespace
                api.create_namespaced_service(namespace=namespace, body=service)
                logger.info(f'Service {service_name} created.')
            except ApiException as e:
                logger.error(f'Error creating service: {e}')

        logger.info(f'Finished creating Services for simulation {name} on graph {graph_name}.')

    # get simulation settings from the spec
    repetitions = spec.get('repetitions', 1)
    visualize = spec.get('visualize', False)
    simulationProperties = spec.get('simulationProperties', {})
    # get graph settings frrom the graph spec
    graphType = graph_spec.get('graphType', 'normal')
    modularity = graph_spec.get('modularity', None)
    graphProperties = graph_spec.get('graphProperties', {})

    def create_runner_pod():
        """
        Create a runner pod for the simulation.

        Returns:
            None
        """
        # create the runner pod
        pod_name = f'{name}-runner'

        labels = {
            'app': 'gossip',
            'simulation': name,
            'graph': graph_name,
            'node': 'runner'
        }

        # string representation of all created pods
        nodes_str = ','.join(pods)

        env = []
        # set all necessary environment variables
        env.append(client.V1EnvVar(name=ENVIRONMENT_SIMULATION, value=name))
        env.append(client.V1EnvVar(name=ENVIRONMENT_ALGORITHM, value=algorithm))
        env.append(client.V1EnvVar(name=ENVIRONMENT_REPETITIONS, value=str(repetitions)))
        env.append(client.V1EnvVar(name=ENVIRONMENT_ADJ_LIST, value=str_adj_list))
        env.append(client.V1EnvVar(name=ENVIRONMENT_NODES, value=nodes_str))

        # node communities specific environment variable
        # set for graph highlighting in plots
        if algorithm in NODE_COMMUNITIES_SET:
            node_community_string = json.dumps(node_community_dict)
            env.append(client.V1EnvVar(name=ENVIRONMENT_NODE_COMMUNITIES, value=node_community_string))

        env.append(client.V1EnvVar(name=ENVIRONMENT_VISUALIZE, value=str(visualize)))

        # simulation properties for logging 
        simulation_properties = simulationProperties.copy()
        simulation_properties_string = json.dumps(simulation_properties)
        env.append(client.V1EnvVar(name=ENVIRONMENT_SIMULATION_PROPERTIES, value=simulation_properties_string))
        # graph properties for logging
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

        # define the runner pod
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
            # create the runner pod
            api.create_namespaced_pod(namespace=namespace, body=pod)
            logger.info(f'Pod {pod_name} created')
        except ApiException as e:
            logger.error(f'Error creating pod: {e}')

        logger.info(f'Finished creating simulation runner pod for simulation {name}.')

    def create_runner_service():
        """
        Create a runner service for the simulation.

        Returns:
            None
        """
        # create the runner service
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

        # define the runner service
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
            # create the runner service
            api.create_namespaced_service(namespace=namespace, body=service)
            logger.info(f'Service {service_name} created')
        except ApiException as e:
            logger.error(f'Error creating service: {e}')

        logger.info(f'Finished creating simulation runner service for simulation {name}.')

    # create services
    create_node_services()
    create_runner_service()
    # create node pods
    create_node_pods()

    # Set the labels used for selecting created node pods
    labels = {
        'app': 'gossip',
        'simulation': name,
        'graph': graph_name
    }

    # wait until all node pods started before the runner pod is started
    # this is done to prevent runner pod restarts 
    # restarts can happen because the nodes need to be running for communication purposes
    logger.info(f'Waiting for node pods to start...')
    while True:
        # List all pods matching the specified labels
        node_pods = api.list_namespaced_pod(namespace=namespace,
                                            label_selector=','.join([f"{k}={v}" for k, v in labels.items()]))

        # Check if all pods are in the "Running" state
        if all(pod.status.phase == 'Running' for pod in node_pods.items):
            logger.info('All node pods are now running.')
            # All pods are running, exit the loop
            break
            # Wait for 1 second before checking again
        time.sleep(1)

        # All pods are running, create the runner pod
    create_runner_pod()

    logger.info(f'Finished creating resources for simulation {name}.')


@kopf.on.delete('gossip.io', 'v1', 'simulations')
def delete_services_and_pods(body, **kwargs):
    """
    Delete services and pods associated with a simulation.

    Args:
        body (dict): The request body containing the information about the simulation to delete.
        **kwargs: Additional keyword arguments.

    Returns:
        None
    """
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
