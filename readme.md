# Simulation Operator

The **Gossip Simulation** is based on a Kubernetes operator built with Python.
Its purpose is simulating different gossip protocols on variable network graphs.
The simulations run in a Kubernetes cluster. 
This README provides an overview of the project, how to build the Docker image, deploy it, and use it to manage simulations.

## Table of Contents
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Minikube](#minikube)
- [Building the Docker Images](#building-the-docker-images)
- [Deploying the Object Storage](#deploying-the-object-storage)
- [Deploying the Operator](#deploying-the-operator)
- [Usage](#usage)
- [Customization](#customization)

## Architecture

## Prerequisites

Before you begin, ensure you have the following prerequisites:
- Docker installed on your local machine
- Access to a Kubernetes cluster
- `kubectl` configured to communicate with your cluster

## Minikube

1. Install Minikube by following the installation instructions in the [official Minikube documentation](https://minikube.sigs.k8s.io/docs/start/).

2. Start a Minikube cluster by running the following command:

   ```bash
   minikube start --extra-config=kubelet.max-pods=1000
   ```
   
   This increases the max pod size, which is recommended to run this application.
   The pods are very lightweight.

3. Verify that Minikube is running and the cluster is healthy:

   ```bash
   minikube status
   ```
   You should see output indicating that the cluster is running.

4. You can now use Minikube to deploy and manage Kubernetes resources locally for development and testing purposes.
   With minikube only small simulations can be run.
   The accurate number is dependent on the hardware resources of the local machine.

## Building the Docker Images

To build the Docker images for the simulation environment, follow these steps:

1. Clone this repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
   
2. Build the images
   
   ```bash
   docker build -t user/node-app:latest .\Dockerfile.node
   
   docker build -t user/runner-app:latest .\Dockerfile.runner
   
   docker build -t user/simulation-operator:latest .\Dockerfile.operator
   ```
   
3. Push the images

   ```bash
   docker push user/node-app:latest 
   
   docker push user/runner-app:latest
   
   docker push user/simulation-operator:latest
   ```
   
   In case a multi-architecture image is needed use to build and directly push:
   ```bash
   docker buildx build --builder=eager_aryabhata --platform linux/amd64,linux/arm64 -t user/node-app --push:latest -f ./Dockerfile.node .
   
   docker buildx build --builder=eager_aryabhata --platform linux/amd64,linux/arm64 -t user/runner-app:latest --push -f ./Dockerfile.runner .
   
   docker buildx build --builder=eager_aryabhata --platform linux/amd64,linux/arm64 -t user/simulation-operator:latest --push -f ./Dockerfile.operator .
   ```
   If building images for a private gossip registry that is secured, the certificates can be copied to the build container:
   ```bash
   docker cp .\my_ca.crt buildx_buildkit_eager_aryabhata0:/usr/local/share/ca-certificates/my_ca.crt
   ```
   
## Deploying the Object Storage

For deploying the object storage, please refer to the instructions provided in the [Object Storage README](./minio/readme.md).


## Deploying the CustomResourceDefinitions


## Deploying the Operator

To deploy the operator, follow the steps outlined in the [Operator README](./operator/readme.md).

## Running a Simulation


## Customization
You can customize the operator by modifying the python scripts and any other relevant files. 
If new algorithms are to be added, this can be done by extending the node application.
Additionally, you can customize the operator's behavior by adjusting the Kubernetes custom resource definitions (CRD).
   
   