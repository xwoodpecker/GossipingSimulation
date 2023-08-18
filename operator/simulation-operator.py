import logging
import math
import time
import json
import kopf
import uuid
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import networkx as nx
import community as community_louvain
from networkx import PowerIterationFailedConvergence

from cfg import *

# Create a custom logger
log = logging.getLogger(__name__)

# Set the logging level
log.setLevel(logging.INFO)

# Create a formatter with the desired log message format
formatter = logging.Formatter('%(levelname)s:%(message)s')

# Create a handler and set the formatter
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Add the handler to the logger
log.addHandler(handler)

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
    simulation_id = str(uuid.uuid4())
    graph_selector = spec.get('graphSelector', {})
    match_labels = graph_selector.get('matchLabels', {})

    graph_dict = {}  # Dictionary to store graph name and graph spec

    if 'name' in match_labels:
        series_simulation = False
        # Select a single graph based on 'name' label
        graph_name = match_labels['name']
        graph_obj = customs_api.get_namespaced_custom_object('gossip.io', 'v1', namespace, 'graphs', graph_name)
        graph_dict[graph_name] = graph_obj['spec']
        log.info(f"Selected graph '{graph_name}' from graphSelector.")
    elif 'series' in match_labels:
        series_simulation = True
        # Select multiple graphs based on 'series' label
        series_name = match_labels['series']
        # Retrieve all graphs in the series based on the 'series' label
        graph_objs = customs_api.list_namespaced_custom_object('gossip.io', 'v1', namespace, 'graphs',
                                                               label_selector=f"series={series_name}")
        for graph_obj in graph_objs['items']:
            graph_name = graph_obj['metadata']['name']
            graph_dict[graph_name] = graph_obj['spec']
        log.info(f"Selected graphs in series '{series_name}' from graphSelector.")

    # Perform simulation for each selected graph spec
    for graph_index, (graph_name, graph_spec) in enumerate(graph_dict.items()):
        log.info(f'Creating simulation for graph {graph_name}.')
        is_last_graph_spec = (graph_index == len(graph_dict) - 1)
        # Convert the adjacency list from a comma-separated string to a list of tuples
        split_adj_list = graph_spec.get('adjacencyList', '')
        str_adj_list = ''.join(split_adj_list)
        split_adj_list = [split_str.rstrip(',') for split_str in split_adj_list]
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

        log.info(f'Neighbors of each node: {neighbors}')

        algorithm = spec.get('algorithm', DEFAULT_ALGORITHM)
        repetitions = spec.get('repetitions', 1)
        log.info(f'Simulation running algorithm {algorithm}')

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
            log.info(f'Node communities: {node_community_dict}')
            return node_community_dict, community_node_dict

        # communities are needed for weighted_factor and community probability assignment
        if algorithm in NODE_COMMUNITIES_SET:
            graph = nx.parse_adjlist(split_adj_list)
            # apply louvain method on the graph
            partition = community_louvain.best_partition(graph)
            partition = {int(k): int(v) for k, v in partition.items()}
            node_community_dict, community_node_dict = get_community_node_dict(partition)

            # weighted factor algorithms use a factor to modify the probability
            # of selecting a partner inside or outside the community
            if algorithm in WEIGHTED_FACTOR_SET:
                factors = spec.get('factor', [DEFAULT_FACTOR])


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

            if algorithm in ADVANCED_CLUSTERING_SET:
                weighting_params_a = spec.get('weightingParamA', [DEFAULT_WEIGHTING_PARAM])

                if algorithm in BETWEENNESS_SET:
                    betweenness_centralities = nx.betweenness_centrality(graph)

                if algorithm in EIGENVECTOR_SET:
                    try:
                        eigenvector_centralities = nx.eigenvector_centrality(graph, max_iter=1000)
                    except PowerIterationFailedConvergence:
                        log.error('Could not compute eigenvector centralities. Aborting...')
                        try:
                            customs_api.delete_namespaced_custom_object(
                                namespace=namespace,
                                name=name
                            )
                            print(f"Custom resource {name} deleted successfully.")
                        except client.ApiException as e:
                            print(f"Error deleting custom resource {name}: {e}")
                        return

                if algorithm in HUB_SCORE_SET:
                    hub_scores, authority_scores = nx.hits(graph)


        # memory algorithms use a factor to modify the probability
        # of selecting a partner that has already been selected in a previous gossiping
        if algorithm in MEMORY_SET:
            prior_partner_factors = spec.get('priorPartnerFactor', [DEFAULT_PRIOR_PARTNER_FACTOR])

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

        def get_resource_name(graph_index, name, node):
            """
            Generate a resource name based on the simulation name and node ID.

            Args:
                name (str): The name of the simulation.
                node (str): The node ID.

            Returns:
                str: The generated resource name.
            """
            return f'{name}-g{graph_index}-n{node}'

        def create_node_pods():
            """
            Create node pods for the simulation.

            Returns:
                None
            """
            batch_size = CREATE_POD_BATCH_SIZE
            num_nodes = len(nodes)
            num_batches = math.ceil(num_nodes / batch_size)

            # Create a Pod for each node in the graph
            for batch_index in range(num_batches):
                start_index = batch_index * batch_size
                end_index = min((batch_index + 1) * batch_size, num_nodes)
                batch_nodes = nodes[start_index:end_index]

                for node in batch_nodes:
                    # Create a Pod for this node
                    pod_name = get_resource_name(graph_index, name, node)

                    labels = {
                        'app': 'gossip',
                        'simulation': name,
                        'simulation_id': simulation_id,
                        'graph': graph_name,
                        'node': str(node)
                    }

                    env = []

                    neighbor_nodes = neighbors[node]

                    # set environment variables
                    neighbors_str = ','.join([get_resource_name(graph_index, name, n) for n in neighbor_nodes])
                    env.append(client.V1EnvVar(name=ENVIRONMENT_NEIGHBORS, value=neighbors_str))
                    env.append(client.V1EnvVar(name=ENVIRONMENT_ALGORITHM, value=algorithm))
                    env.append(client.V1EnvVar(name=ENVIRONMENT_REPETITIONS, value=str(repetitions)))
                    env.append(client.V1EnvVar(name=ENVIRONMENT_RANDOM_INITIALIZATION, value=str(randomInitialization)))
                    if not randomInitialization:
                        env.append(client.V1EnvVar(name=ENVIRONMENT_NODE_VALUE, value=str(node_values[node])))

                    # weighted factor algorithm specific environment variables
                    if algorithm in WEIGHTED_FACTOR_SET:
                        # set the community neighbors of the current node
                        community_id = node_community_dict[node]
                        community_nodes = community_node_dict[community_id]
                        # extract community and non-community neighbors
                        community_neighbors = []
                        # non_community_neighbors = []
                        for neighbor_node in neighbor_nodes:
                            if neighbor_node in community_nodes:
                                community_neighbors.append(neighbor_node)

                        community_neighbors_str = ','.join(
                            [get_resource_name(graph_index, name, n) for n in community_neighbors]
                        )
                        env.append(client.V1EnvVar(name=ENVIRONMENT_COMMUNITY_NEIGHBORS, value=community_neighbors_str))
                        env.append(client.V1EnvVar(name=ENVIRONMENT_FACTOR,
                                                   value=','.join(str(factor) for factor in factors)))

                    # community probabilities algorithm specific environment variables
                    if algorithm in COMMUNITY_PROBABILITIES_SET:
                        # set the same community probabilities of the neighbors for the current node
                        community_id = node_community_dict[node]

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

                    if algorithm in COMMUNITY_BASED_SET:
                        neighboring_communities = []
                        for neighbor in neighbor_nodes:
                            neighbor_community = node_community_dict[neighbor]
                            neighboring_communities.append(str(neighbor_community))
                        neighboring_communities_str = ','.join(neighboring_communities)
                        env.append(client.V1EnvVar(name=ENVIRONMENT_NEIGHBORING_COMMUNITIES,
                                                   value=neighboring_communities_str))

                    def get_data_for_neighbors(dictionary, neighbor_list):
                        neighbor_data_list = []
                        for n in neighbor_list:
                            neighbor_data = dictionary[str(n)]
                            neighbor_data_list.append(neighbor_data)
                        return neighbor_data_list

                    if algorithm in ADVANCED_CLUSTERING_SET:
                        env.append(client.V1EnvVar(name=ENVIRONMENT_WEIGHTING_PARAM_A,
                                                   value=','.join(str(param) for param in weighting_params_a)))

                        if algorithm in BETWEENNESS_SET:
                            betweenness_centralities_neighbors \
                                = get_data_for_neighbors(betweenness_centralities, neighbor_nodes)

                            betweenness_centralities_neighbors_str = ','.join(
                                str(round(item, ADVANCED_ALGORITHM_WEIGHT_ROUNDING))
                                for item
                                in betweenness_centralities_neighbors
                            )
                            env.append(client.V1EnvVar(name=ENVIRONMENT_BETWEENNESS_CENTRALITIES_NEIGHBORS,
                                                       value=betweenness_centralities_neighbors_str))

                        if algorithm in EIGENVECTOR_SET:
                            eigenvector_centralities_neighbors \
                                = get_data_for_neighbors(eigenvector_centralities, neighbor_nodes)

                            eigenvector_centralities_neighbors_str = ','.join(
                                str(round(item, ADVANCED_ALGORITHM_WEIGHT_ROUNDING))
                                for item
                                in eigenvector_centralities_neighbors
                            )
                            env.append(client.V1EnvVar(name=ENVIRONMENT_EIGENVECTOR_CENTRALITIES_NEIGHBORS,
                                                       value=eigenvector_centralities_neighbors_str))

                        if algorithm in HUB_SCORE_SET:
                            hub_scores_neighbors = get_data_for_neighbors(hub_scores, neighbor_nodes)

                            hub_scores_neighbors_str = ','.join(
                                str(round(item, ADVANCED_ALGORITHM_WEIGHT_ROUNDING))
                                for item
                                in hub_scores_neighbors
                            )
                            env.append(client.V1EnvVar(name=ENVIRONMENT_HUB_SCORES_NEIGHBORS,
                                                       value=hub_scores_neighbors_str))

                    # memory algorithm specific environment variables
                    if algorithm in MEMORY_SET:
                        env.append(client.V1EnvVar(name=ENVIRONMENT_PRIOR_PARTNER_FACTOR,
                                                   value=','.join(str(factor) for factor in prior_partner_factors)))

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
                            containers=[container],
                            image_pull_secrets=[client.V1LocalObjectReference(name=REGISTRY_SECRET_NAME)],
                            topology_spread_constraints=[
                                client.V1TopologySpreadConstraint(
                                    max_skew=1,
                                    topology_key="simulation_node",
                                    when_unsatisfiable="DoNotSchedule",
                                    label_selector=client.V1LabelSelector(
                                        match_labels={
                                            "app": "gossip"
                                        }
                                    )
                                )
                            ]
                        )
                    )
                    try:
                        # create the pod in the current namespace
                        api.create_namespaced_pod(namespace=namespace, body=pod)
                        log.info(f'Pod {pod_name} created.')
                        pods.append(pod_name)
                    except ApiException as e:
                        log.error(f'Error creating pod: {e}')

                # Wait for the pods in this batch to start
                log.info(f'Waiting for node pods in batch {batch_index + 1}/{num_batches} to start...')
                while True:
                    # List all pods matching the specified labels
                    node_pods = api.list_namespaced_pod(namespace=namespace, label_selector=','.join(
                        [f"{k}={v}" for k, v in labels.items()]))

                    # Check if all pods in this batch are in the "Running" state
                    if all(pod.status.phase == 'Running' for pod in node_pods.items):
                        log.info(f'All node pods in batch {batch_index + 1}/{num_batches} are now running.')
                        break

            log.info(f'Finished creating Pods for simulation {name} on graph {graph_name}.')

        def create_node_services():
            """
            Create pods for the simulation.

            Returns:
                None
            """
            for node in nodes:
                # Create a Service for this node
                service_name = get_resource_name(graph_index, name, node)

                labels = {
                    'app': 'gossip',
                    'simulation': name,
                    'simulation_id': simulation_id,
                    'graph': graph_name,
                    'node': str(node)
                }
                selector = {
                    'app': 'gossip',
                    'simulation': name,
                    'simulation_id': simulation_id,
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
                    log.info(f'Service {service_name} created.')
                except ApiException as e:
                    log.error(f'Error creating service: {e}')

            log.info(f'Finished creating Services for simulation {name} on graph {graph_name}.')

        # get simulation settings from the spec
        visualize = spec.get('visualize', False)
        simulationProperties = spec.get('simulationProperties', {})
        # get graph settings from the graph spec
        graphType = graph_spec.get('graphType', 'undefined')
        graphProperties = graph_spec.get('graphProperties', {})

        def create_runner_pod():
            """
            Create a runner pod for the simulation.

            Returns:
                None
            """
            # create the runner pod
            pod_name = f'{name}-g{graph_index}-runner'

            labels = {
                'app': 'gossip',
                'simulation': name,
                'simulation_id': simulation_id,
                'graph': graph_name,
                'node': 'runner'
            }

            # string representation of all created pods
            nodes_str = ','.join(pods)

            env = []
            # set all necessary environment variables
            env.append(client.V1EnvVar(name=ENVIRONMENT_SIMULATION, value=name))
            env.append(client.V1EnvVar(name=ENVIRONMENT_SIMULATION_ID, value=simulation_id))
            env.append(client.V1EnvVar(name=ENVIRONMENT_SERIES_SIMULATION, value=str(series_simulation)))
            env.append(client.V1EnvVar(name=ENVIRONMENT_GRAPH_NAME, value=graph_name))
            env.append(client.V1EnvVar(name=ENVIRONMENT_ALGORITHM, value=algorithm))
            env.append(client.V1EnvVar(name=ENVIRONMENT_REPETITIONS, value=str(repetitions)))
            env.append(client.V1EnvVar(name=ENVIRONMENT_ADJ_LIST, value=str_adj_list))
            env.append(client.V1EnvVar(name=ENVIRONMENT_NODES, value=nodes_str))

            # node communities specific environment variable
            # set for graph highlighting in plots
            if algorithm in NODE_COMMUNITIES_SET:
                node_community_string = json.dumps(node_community_dict)
                env.append(client.V1EnvVar(name=ENVIRONMENT_NODE_COMMUNITIES, value=node_community_string))

                if algorithm in WEIGHTED_FACTOR_SET:
                    env.append(client.V1EnvVar(name=ENVIRONMENT_FACTOR,
                                               value=','.join(str(factor) for factor in factors)))

            if algorithm in ADVANCED_CLUSTERING_SET:
                env.append(client.V1EnvVar(name=ENVIRONMENT_WEIGHTING_PARAM_A,
                                           value=','.join(str(param) for param in weighting_params_a)))
            if algorithm in MEMORY_SET:
                env.append(client.V1EnvVar(name=ENVIRONMENT_PRIOR_PARTNER_FACTOR,
                                           value=','.join(str(factor) for factor in prior_partner_factors)))

            env.append(client.V1EnvVar(name=ENVIRONMENT_VISUALIZE, value=str(visualize)))

            # simulation properties for logging
            simulation_properties = simulationProperties.copy()
            simulation_properties_string = json.dumps(simulation_properties)
            env.append(client.V1EnvVar(name=ENVIRONMENT_SIMULATION_PROPERTIES, value=simulation_properties_string))
            # graph properties for logging
            graph_properties = graphProperties.copy()
            graph_properties['graphType'] = graphType
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
                    containers=[container],
                    image_pull_secrets=[client.V1LocalObjectReference(name=REGISTRY_SECRET_NAME)]
                )
            )
            try:
                # create the runner pod
                api.create_namespaced_pod(namespace=namespace, body=pod)
                log.info(f'Pod {pod_name} created.')
            except ApiException as e:
                log.error(f'Error creating pod: {e}')

            log.info(f'Finished creating simulation runner pod for simulation {name}.')

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
                'simulation_id': simulation_id,
                'graph': graph_name,
                'node': 'runner'
            }
            selector = {
                'app': 'gossip',
                'simulation': name,
                'simulation_id': simulation_id,
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
                log.info(f'Service {service_name} created')
            except ApiException as e:
                log.error(f'Error creating service: {e}')

            log.info(f'Finished creating simulation runner service for simulation {name}.')

        # create services
        create_node_services()
        create_runner_service()
        # create node pods
        create_node_pods()

        # Set the labels used for selecting created node pods
        labels = {
            'app': 'gossip',
            'simulation': name,
            'simulation_id': simulation_id,
            'graph': graph_name
        }

        # wait until all node pods started before the runner pod is started
        # this is done to prevent runner pod restarts
        # restarts can happen because the nodes need to be running for communication purposes
        log.info(f'Waiting for node pods to start...')
        while True:
            # List all pods matching the specified labels
            node_pods = api.list_namespaced_pod(namespace=namespace,
                                                label_selector=','.join([f"{k}={v}" for k, v in labels.items()]))

            # Check if all pods are in the "Running" state
            if all(pod.status.phase == 'Running' for pod in node_pods.items):
                log.info('All node pods are now running.')
                # All pods are running, exit the loop
                break

            # Wait for 1 second before checking again
            time.sleep(1)

        # All pods are running, create the runner pod
        create_runner_pod()

        log.info(f'Finished creating resources for simulation {name}.')

        # if multiple graphs are to be simulated delete all resources after completion
        if not is_last_graph_spec:
            log.info('Waiting for Simulation to complete...')

            iteration_count = 0
            while True:
                try:
                    pods = api.list_namespaced_pod(namespace=namespace, label_selector=f"simulation={name}")
                    completed_count = sum(1 for pod in pods.items if pod.status.phase == 'Succeeded')
                    total_count = len(pods.items)
                    if completed_count == total_count:
                        log.info(f'Simulation completed for graph {graph_name}.')
                        break
                    elif math.isclose(completed_count / total_count, 0.99, rel_tol=1e-3) and iteration_count >= 10:
                        log.info(f'99% of the pods have completed for graph {graph_name}.')
                        log.info(f'One Pod possibly stuck during container stoppage. Proceeding nonetheless...')
                        break
                    else:
                        # Wait for 2 seconds before checking again
                        time.sleep(2)
                        iteration_count += 1
                except ApiException as e:
                    if e.status == 404:
                        # Pods not found, simulation may not have started yet
                        time.sleep(5)
                    else:
                        raise e

            log.info('Cleaning up simulation before starting the next.')
            log.info('Deleting pods...')
            # Delete all pods with the simulation name label
            for pod in pods.items:
                api.delete_namespaced_pod(pod.metadata.name, namespace)

            log.info('Deleting services...')
            # Delete all services with the simulation name label
            services = api.list_namespaced_service(namespace, label_selector=f"simulation={name}")
            for service in services.items:
                api.delete_namespaced_service(service.metadata.name, namespace)



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

    log.info('Deleting services...')
    # Delete all services with the simulation name label
    services = api.list_namespaced_service(namespace, label_selector=f"simulation={simulation_name}")
    for service in services.items:
        api.delete_namespaced_service(service.metadata.name, namespace)

    log.info('Deleting pods...')
    # Delete all pods with the simulation name label
    pods = api.list_namespaced_pod(namespace, label_selector=f"simulation={simulation_name}")
    for pod in pods.items:
        api.delete_namespaced_pod(pod.metadata.name, namespace)

    log.info(f'Deleted all resources for simulation {simulation_name}.')
