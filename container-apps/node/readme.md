# Gossip Service
## Overview
The Gossip Node Service is a Python script designed to run in a Docker container inside a Kubernetes (K8s) cluster.
It is part of a distributed system where nodes communicate with each other over TCP.
Each Node Service runs inside a Docker container initiated by the "Simulation Operator" within a Kubernetes (K8s) cluster.
The Simulation Operator creates one service pod for each node in a network graph. 
The node is responsible for initializing and managing various gossip algorithms.
The service contains a gRPC server.
Its methods are invoked by the "Simulation Runner".
The Runner is responsible for managing the simulation.
This README provides an overview of the script, its purpose, and instructions for usage.

Key Features:
- Dynamic algorithm selection based on configuration.
- Customizable algorithm parameters.
- Graceful shutdown using a stop event.

## Prerequisites
Before using this script within a K8s cluster, ensure you have the following prerequisites:

- Kubernetes Cluster
- Minio server (for result storage)
- Gossip Runner (Running gRPC client for communication)
- Docker (for containerization)
- Required Python libraries (specified in the script)

The [../proto/gossip.proto](../proto/gossip.proto) file defines the gRPC service and message types for communication between nodes. 
Ensure that the generated code for gRPC communication is used.

The client code generation is done by executing the command:
```bash
python -m grpc_tools.protoc -Iproto --python_out=. --grpc_python_out=. proto/gossip.proto
```

## Usage
To run this script within a K8s cluster, follow these steps:

1. Deploy the "Simulation Operator" within your Kubernetes cluster.
This operator is responsible for initiating and managing the simulation containers.

2. Create a K8s Custom Resource (CR) or YAML manifest to specify the simulation parameters. 
Ensure that you define the necessary environment variables, such as MINIO_ENDPOINT, MINIO_USER, MINIO_PASSWORD, and other simulation-specific variables.

3. Apply the CR or YAML manifest to start the simulation. The "Simulation Operator" will create a Docker container using this script, passing the environment variables and configurations.

4. The script will initiate the gossip simulation within the Docker container.
The results will be stored in the configured Minio bucket within the K8s cluster.

## Algorithm Parameters
You can customize various algorithm parameters by defining them within the K8s Custom Resource or YAML manifest. These parameters include the algorithm used, the number of repetitions, visualization settings, and more. Refer to the script comments for details on available parameters.
