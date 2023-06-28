ENVIRONMENT_HOSTNAME = "HOSTNAME"
ENVIRONMENT_NEIGHBORS = "NEIGHBORS"
ENVIRONMENT_ALGORITHM = "ALGORITHM"
ENVIRONMENT_RANDOM_INITIALIZATION = "RANDOM_INITIALIZATION"
ENVIRONMENT_NODE_VALUE = "NODE_VALUE"
ENVIRONMENT_COMMUNITY_NEIGHBORS = "COMMUNITY_NEIGHBORS"
ENVIRONMENT_FACTOR = "FACTOR"
ENVIRONMENT_SAME_COMMUNITY_PROBABILITIES_NEIGHBORS = "SAME_COMMUNITY_PROBABILITIES_NEIGHBORS"
ENVIRONMENT_PRIOR_PARTNER_FACTOR = "PRIOR_PARTNER_FACTOR"
ENVIRONMENT_MINIO_ENDPOINT = "MINIO_ENDPOINT"
ENVIRONMENT_MINIO_USER = "MINIO_USER"
ENVIRONMENT_MINIO_PASSWORD = "MINIO_PASSWORD"
ENVIRONMENT_SIMULATION = "SIMULATION"
ENVIRONMENT_GRAPH_NAME = "GRAPH_NAME"
ENVIRONMENT_SERIES_SIMULATION = "SERIES_SIMULATION"
ENVIRONMENT_NODE_COMMUNITIES = "NODE_COMMUNITIES"
ENVIRONMENT_REPETITIONS = "REPETITIONS"
ENVIRONMENT_SIMULATION_PROPERTIES = "SIMULATION_PROPERTIES"
ENVIRONMENT_GRAPH_PROPERTIES = "GRAPH_PROPERTIES"
ENVIRONMENT_ADJ_LIST = "ADJ_LIST"
ENVIRONMENT_NODES = "NODES"
ENVIRONMENT_VISUALIZE = "VISUALIZE"

DEFAULT_ALGORITHM = "default"
DEFAULT_FACTOR = 1.5
DEFAULT_PRIOR_PARTNER_FACTOR = 0.5
DEFAULT_SIMULATION_NAME = "unidentified"
DEFAULT_REPETITIONS = 1
DEFAULT_GRAPH_NAME = 'unspecified'

ALGORITHM_DEFAULT_MEMORY = "default_memory"
ALGORITHM_DEFAULT_COMPLEX_MEMORY = "default_complex_memory"
ALGORITHM_WEIGHTED_FACTOR = "weighted_factor"
ALGORITHM_WEIGHTED_FACTOR_MEMORY = "weighted_factor_memory"
ALGORITHM_WEIGHTED_FACTOR_COMPLEX_MEMORY = "weighted_factor_complex_memory"
ALGORITHM_COMMUNITY_PROBABILITIES = "community_probabilities"
ALGORITHM_COMMUNITY_PROBABILITIES_MEMORY = "community_probabilities_memory"
ALGORITHM_COMMUNITY_PROBABILITIES_COMPLEX_MEMORY = "community_probabilities_complex_memory"

DEFAULT_SET = DEFAULT_ALGORITHM + ALGORITHM_DEFAULT_MEMORY + ALGORITHM_DEFAULT_COMPLEX_MEMORY 
WEIGHTED_FACTOR_SET = (ALGORITHM_WEIGHTED_FACTOR, ALGORITHM_WEIGHTED_FACTOR_MEMORY, ALGORITHM_WEIGHTED_FACTOR_COMPLEX_MEMORY)
COMMUNITY_PROBABILITIES_SET = (ALGORITHM_COMMUNITY_PROBABILITIES, ALGORITHM_COMMUNITY_PROBABILITIES_MEMORY, ALGORITHM_COMMUNITY_PROBABILITIES_COMPLEX_MEMORY)
NODE_COMMUNITIES_SET = WEIGHTED_FACTOR_SET + COMMUNITY_PROBABILITIES_SET
COMPLEX_MEMORY_SET = (ALGORITHM_DEFAULT_COMPLEX_MEMORY, ALGORITHM_WEIGHTED_FACTOR_COMPLEX_MEMORY, ALGORITHM_COMMUNITY_PROBABILITIES_COMPLEX_MEMORY)
MEMORY_SET = (ALGORITHM_DEFAULT_MEMORY, ALGORITHM_WEIGHTED_FACTOR_MEMORY, ALGORITHM_COMMUNITY_PROBABILITIES_MEMORY) + COMPLEX_MEMORY_SET

GRPC_SERVICE_PORT = 50051
TCP_SERVICE_PORT = 90
TCP_BUFSIZE = 1024

MINIO_BUCKET_NAME = "simulations"
MINIO_SIMULATIONS_CSV_NAME = "simulation-results.csv"
TMP_CSV_NAME = "simulations-results_tmp.csv"
MINIO_TIME_FORMAT_STRING = "%Y-%m-%d-%H-%M"
RESULT_TIME_FORMAT_STRING = "%Y-%m-%d-%H-%M-%S"

REGEX_NODE_NAME_PATTERN = r"-n(\d+)$"

MAXIMUM_NODE_NUMBER_NORMAL_PLOT = 100
DEFAULT_NODE_COLOR = '#FFFFFF'

COMMUNITY_PROBABILITIES_ROUNDING = 4

REGISTRY_SECRET_NAME = "reg-cred-secret"
#DOCKER_NODE_IMAGE = "xwoodpecker/node-example:latest"
DOCKER_NODE_IMAGE = "gossip-registry:32652/node-example:sim_v2"
DOCKER_NODE_NAME = "node-example"
#DOCKER_RUNNER_IMAGE = "xwoodpecker/runner-example:latest"
DOCKER_RUNNER_IMAGE = "gossip-registry:32652/runner-example:sim_v2"
DOCKER_RUNNER_NAME = "runner-example"

MINIO_CONFIGMAP_NAME = "minio-configmap"
MINIO_SECRETS_NAME = "minio-secrets"

GRAPH_TYPE_SIMPLE_SHORT = "simpl"
GRAPH_TYPE_COMPLEX_SHORT = "compl"
GRAPH_TYPE_SCALE_FREE_SHORT = "scale"
GRAPH_TYPE_BARABASI_ALBERT_SHORT = "barab"
GRAPH_TYPE_HOLME_KIM_SHORT = "holme"

GRAPH_TYPE_SIMPLE = "simple-own-implementation"
GRAPH_TYPE_COMPLEX = "complex-fixed-modularity"
GRAPH_TYPE_SCALE_FREE = "scale-free"
GRAPH_TYPE_BARABASI_ALBERT = "barabasi-albert"
GRAPH_TYPE_HOLME_KIM = "holme-kim"

POISSON_DISTRIBUTION_NAME = 'poisson'
REGULAR_DISTRIBUTION_NAME = 'regular'
GEOMETRIC_DISTRIBUTION_NAME = 'geometric'
SCALE_FREE_DISTRIBUTION_NAME = 'scale-free'



