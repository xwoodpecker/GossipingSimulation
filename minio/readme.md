# Deploying Minio

This README section outlines the steps to deploy Minio in your Kubernetes cluster. 
Ensure you have the necessary prerequisites in place before proceeding.

## Prerequisites

Before deploying Minio, ensure you have the following prerequisites:

- A running Kubernetes cluster (e.g., Minikube or your own cluster)
- `kubectl` configured to communicate with your cluster
- Helm installed (for custom installation)

## (Optional) Custom Configuration for Helm

The following Kubernetes resources can be used for customizing the Helm deployment process:
   
1. minio-pv.yaml

    Creates the PersistentVolume
    ```bash
    kubectl apply -f minio-pv.yaml
    ```


2. minio-pvc.yaml

    Creates the PersistentVolumeClaim
    ```bash
    kubectl apply -f minio-pvc.yaml
    ```

The created PersistentVolume and PersistentVolumeClaim can be manually provisioned when deploying the helm chart.

## Deployment with Helm
Either deploy the minio helm chart normally or use the command below to claim an existing PersistentVolume.

```bash
helm install minio-custom --set persistence.existingClaim=minio-pvc --set volumePermissions.enabled=true oci://registry-1.docker.io/bitnamicharts/minio
```

## Deployment Configuration 
Obtain credentials by running:
```bash
kubectl get secret --namespace default minio -o jsonpath="{.data.root-user}"
```
Creates the ConfigMap and Secret:
 ```bash
 kubectl apply -f config.yaml
 ```


To expose the Minio service externally, you can edit the service to use a NodePort. Run the following command to edit the service:
```bash 
kubectl edit svc my-minio
```
Change the service type to NodePort, and add the desired NodePort(s).

Obtain the Minio service URL from your minikube :
```bash 
minikube service minio --url
```
