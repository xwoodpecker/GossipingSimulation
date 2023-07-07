import os
import json
import sqlite3
import warnings

import pandas as pd

import numpy as np
from minio import Minio
from minio.error import S3Error
from cfg import MINIO_BUCKET_NAME

# Filter and ignore FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

minio_endpoint = "49.13.19.250:32650"
minio_access_key = "admin"
minio_secret_key = "S4Pek7B8gn"

bucket_name = MINIO_BUCKET_NAME
name_contains = 'averaged_result'

# MinIO connection settings
minio_client = Minio(
    minio_endpoint,
    access_key=minio_access_key,
    secret_key=minio_secret_key,
    secure=False
)


def download_files_with_name(path, created_after=None):
    files = dict()
    try:
        # Retrieve list of objects in the bucket
        objects = minio_client.list_objects(bucket_name, recursive=True)

        i = 0
        # Iterate over the objects and download files containing the specified name
        for obj in objects:
            if name_contains in obj.object_name and obj.is_dir is False:
                # Check if the object was created after the specified time
                if created_after and obj.last_modified < created_after:
                    continue
                file_path = obj.object_name
                base_path = os.path.abspath(path)
                file_name = os.path.basename(file_path)
                file_name, file_extension = os.path.splitext(file_name)
                new_file_name = f"{file_name}_{i}{file_extension}"
                destination_path = os.path.join(base_path, new_file_name)
                minio_client.fget_object(bucket_name, file_path, destination_path)
                print(f"Downloaded: {file_path} -> {destination_path}")
                i += 1
                files[destination_path] = file_path
    except S3Error as err:
        print(f"MinIO Error: {err}")

    return files


def set_evaluated_tag(objects):
    # Iterate over each object
    for obj in objects:
        # Get the object's metadata
        obj_metadata = minio_client.stat_object(bucket_name, obj.object_name)

        # Add or update the "evaluated" tag in the metadata
        if 'metadata' not in obj_metadata:
            obj_metadata['metadata'] = {}

        obj_metadata['metadata']['evaluated'] = 'true'

        # Set the updated metadata for the object
        minio_client.set_object_metadata(bucket_name, obj.object_name, obj_metadata)


def read_json_files(directory):
    data_dict = dict()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json') and 'averaged_result' in file:
                file_path = os.path.join(root, file)
                file_path = os.path.abspath(file_path)
                with open(file_path, 'r') as json_file:
                    try:
                        data = json.load(json_file)
                        # Process the JSON data here
                        data_dict[file_path] = data
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON file '{file_path}': {e}")
    return data_dict


def unwrap_graph_metadata(data):
    graph_metadata = data['graph_metadata']
    del data['graph_metadata']
    unwrapped_data = {**data, **graph_metadata}
    return unwrapped_data


def add_object_storage_path(object_storage_path, data):
    data['objectStoragePath'] = object_storage_path
    return data


def remove_adj_list(data):
    del data['adj_list']
    return data


def rename_num_rounds(data):
    num_rounds_data = data['num_rounds']
    del data['num_rounds']
    data['numRounds'] = num_rounds_data
    return data


def determine_column_type(dtype):
    if dtype == 'int64':
        return 'INTEGER'
    elif dtype == 'float64':
        return 'REAL'
    elif dtype == 'bool':
        return 'INTEGER'  # SQLite does not have a boolean type, use INTEGER instead
    else:
        return 'TEXT'


directory_path = '.\\simulations'
# Create the 'simulations' folder if it doesn't exist
os.makedirs(directory_path, exist_ok=True)

# Create a connection to the SQLite database
conn = sqlite3.connect('results.db')
cursor = conn.cursor()

last_import_date = None

# Insert a single row into the table if it's empty
cursor.execute("SELECT COUNT(*) FROM LAST_IMPORT_DATE")
count = cursor.fetchone()[0]
if count == 0:
    cursor.execute("INSERT INTO LAST_IMPORT_DATE VALUES (datetime('now'))")
else:
    # Retrieve the current timestamp from the table
    cursor.execute("SELECT date FROM LAST_IMPORT_DATE")
    result = cursor.fetchone()
    if result:
        last_import_date = result[0]

file_dict = download_files_with_name(directory_path, last_import_date)
# Usage example
file_data = read_json_files(directory_path)

for directory, data in file_data.items():
    modified_data = unwrap_graph_metadata(data)
    modified_data = add_object_storage_path(file_dict[directory], modified_data)
    modified_data = remove_adj_list(modified_data)
    modified_data = rename_num_rounds(modified_data)
    file_data[directory] = modified_data

file_data_values = list(file_data.values())

# Convert the data into a pandas DataFrame
df = pd.DataFrame(file_data_values)

# Iterate over the columns
for column in df.columns:
    try:
        # Try converting the values to float
        df[column] = df[column].astype(float)
    except ValueError:
        # If conversion throws an exception, change the data type to string/text
        df[column] = df[column].astype(str)

# Get the unique keys from the dictionaries
keys = set(df.columns)

# Save the DataFrame to the database
table_name = 'AVERAGED_RESULTS'

# Extract column names and types from the Pandas DataFrame
column_names = ','.join(df.columns)
column_types = ','.join([df[col].dtype.name for col in df.columns])

# Check if the table already exists
cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
existing_table = cursor.fetchone()

if existing_table:
    # Get the existing column names from the table
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_columns = [column[1] for column in cursor.fetchall()]

    # Find the missing columns
    missing_columns = set(df.columns) - set(existing_columns)

    for column in missing_columns:
        dtype = df[column].dtype
        sql_type = determine_column_type(dtype)
        alter_table_query = f"ALTER TABLE {table_name} ADD COLUMN {column} {sql_type}"
        cursor.execute(alter_table_query)
else:
    # Extract column names and types from the Pandas DataFrame
    columns = []
    for column_name, dtype in df.dtypes.items():
        sql_type = determine_column_type(dtype)
        columns.append(f"{column_name} {sql_type}")

    # Create the table with appropriate column names and types
    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)}, " \
                         f"CONSTRAINT unique_objectStoragePath UNIQUE (ObjectStoragePath))"
    cursor.execute(create_table_query)

# Iterate over each row and insert into the table
for _, row in df.iterrows():
    insert_query = f"INSERT OR IGNORE INTO {table_name} VALUES ({', '.join(['?'] * len(row))});"
    cursor.execute(insert_query, tuple(row))

# Create the table if it doesn't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS LAST_IMPORT_DATE
             (date TEXT)''')

conn.commit()
# Close the database connection
conn.close()
