import os
from minio import Minio
from minio.error import S3Error


def main():
    # Create a client with the MinIO server playground, its access key
    # and secret key.
    client = Minio(
        "192.168.178.58:32650",
        access_key="admin",
        secret_key="ULeZ4zcYI9",
        secure=False
    )

    # Define simulation name and object name
    simulation_name = "my_simulation"
    object_name = f"{simulation_name}/simple_graph.png"

    # Make 'simulations' bucket if not exist.
    found = client.bucket_exists("simulations")
    if not found:
        client.make_bucket("simulations")
    else:
        print("Bucket 'simulations' already exists")

    # Upload 'simple_graph.png' as object name
    # '{simulation_name}/simple_graph.png' to bucket 'simulations'
    file_path = r"C:\Users\WoodPecker\Documents\Privat\HTW-Master\Sem4\gossip\_generated_graphs\simple_graph.png"
    with open(file_path, "rb") as file_data:
        file_size = os.path.getsize(file_path)
        client.put_object(
            "simulations",
            object_name,
            file_data,
            file_size,
            content_type="image/png",
            metadata={"graph_name": "simple_graph"},
        )

    print(
        f"'simple_graph.png' is successfully uploaded as object '{object_name}' to bucket 'simulations'."
    )


if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print("error occurred.", exc)