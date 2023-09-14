# Data Processing Script
This README provides an overview of the Python script `result-reader.py`.
Additionally, the different JupyterNotebooks can be found in this directory.
These evaluate the simulation results which are stored in databases after running the script.
The database files can be found under [./simulation-series](./simulation-series).
This script is designed to process data from the Minio Object Storage, and store it in a SQLite database.

## Prerequisites

Before using the script, make sure you have the following prerequisites in place:

- Python 3.x installed on your system.
- Required Python packages installed. You can install them using pip:

```bash
pip install pandas numpy pytz minio
```

- Access to a Minio Object Storage instance with the necessary credentials (access key and secret key).

## Configuration
Ensure you have the necessary configuration in place within the script:

- Minio Configuration: Update the minio_endpoint, minio_access_key, and minio_secret_key variables with the appropriate Minio configuration.
- Object Storage Bucket: Set the bucket_name variable to the name of the Minio bucket where your data is stored.



## Usage
To use the script, follow these steps:
Open your terminal or command prompt and navigate to the directory containing the script.
Run the script.
The script will connect to your Minio Object Storage.
Afterwards, it will download the simulation results that got stored after the last time the script was run.
The results are then stored in a SQLite database.
The last execution time of the script is also saved.

## Evaluation
The simulation results can then be evaluated.
For the present simulation series this is done with JupyterNotebooks.
These contain data analytics and plots.
Their code structure is quite similar.
They are just used to compare the different gossip algorithms obtain visualisations.
