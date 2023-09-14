# Gossip Runner
## Overview
This Python script is designed to run a gossip simulation on a network graph within a Kubernetes (K8s) cluster. 
It is intended to be initiated by the Simulation Operator within the cluster. 
The script utilizes Minio for storing simulation results and communicates with nodes in the network using gRPC. 
It is highly configurable and can be used for various gossip-based simulations.

## Prerequisites
Before using this script within a K8s cluster, ensure you have the following prerequisites:

- Kubernetes Cluster
- Minio server (for result storage)
- Gossip Service (Running gRPC server for communication with network nodes)
- Docker (for containerization)
- Required Python libraries (specified in the script)

## Usage
To run this script within a K8s cluster, follow these steps:

1. Deploy the "Simulation Operator" within your Kubernetes cluster.
This operator is responsible for initiating and managing the simulation containers.

2. Create a K8s Custom Resource (CR) or YAML manifest to specify the simulation parameters. 
Ensure that you define the necessary environment variables, such as MINIO_ENDPOINT, MINIO_USER, MINIO_PASSWORD, and other simulation-specific variables.

3. Apply the CR or YAML manifest to start the simulation. The "Simulation Operator" will create a Docker container using this script, passing the environment variables and configurations.

4. The script will initiate the gossip simulation within the Docker container. The results will be stored in the configured Minio bucket within the K8s cluster.

## Simulation Parameters
You can customize various simulation parameters by defining them within the K8s Custom Resource or YAML manifest. These parameters include the algorithm used, the number of repetitions, visualization settings, and more. Refer to the script comments for details on available parameters.

## Result Storage
The simulation results, including PNG images, GIFs, GEXF files, and JSON data, will be stored in the specified Minio bucket within the K8s cluster. You can access and analyze the results as needed.