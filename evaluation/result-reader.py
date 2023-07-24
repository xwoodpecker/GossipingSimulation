import os
import json
import sqlite3
import warnings

import pandas as pd

from datetime import datetime
import numpy as np
import pytz
from minio import Minio
from minio.error import S3Error
from cfg import MINIO_BUCKET_NAME

# Filter and ignore FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

minio_endpoint = "49.13.19.250:32650"
minio_access_key = "admin"
minio_secret_key = "S4Pek7B8gn"

bucket_name = MINIO_BUCKET_NAME
name_contains_averaged = '_averaged_result'
name_contains_summary = '_summary'

table_name_averaged_results = 'AVERAGED_RESULTS'
table_name_summaries = 'SUMMARIES'

last_import_date_averaged_results_table_name = 'LAST_IMPORT_DATE_AVERAGED_RESULTS'
last_import_date_summaries_table_name = 'LAST_IMPORT_DATE_SUMMARIES'

# MinIO connection settings
minio_client = Minio(
    minio_endpoint,
    access_key=minio_access_key,
    secret_key=minio_secret_key,
    secure=False
)


def download_files_with_name(name_contains, path, created_after=None):
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


def read_json_files(file_paths):
    data_dict = dict()
    for file_path in file_paths:
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


def remove_redundant(data):
    del data['adj_list']
    data.pop('nodeCommunities', None)
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


def update_last_import_date(last_import_table_name):
    last_import_date = None
    format_string = '%Y-%m-%d %H:%M:%S'
    # Create the table if it doesn't exist
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {last_import_table_name} (date TEXT)")

    # Insert a single row into the table if it's empty
    cursor.execute(f"SELECT COUNT(*) FROM {last_import_table_name}")
    count = cursor.fetchone()[0]
    current_timestamp = datetime.now().strftime(format_string)
    if count == 0:
        cursor.execute(f"INSERT INTO {last_import_table_name} VALUES ('{current_timestamp}')")
    else:
        # Retrieve the current timestamp from the table
        cursor.execute(f"SELECT date FROM {last_import_table_name}")
        result = cursor.fetchone()
        if result:
            last_import_date = datetime.strptime(result[0], format_string)
            timezone = pytz.timezone('Europe/Paris')
            last_import_date = timezone.localize(last_import_date)
            cursor.execute(f"UPDATE {last_import_table_name} SET date = ?", (current_timestamp,))

    return last_import_date


last_import_date_averaged_results = update_last_import_date(last_import_date_averaged_results_table_name)
last_import_date_summaries = update_last_import_date(last_import_date_summaries_table_name)

file_dict_averaged_results = download_files_with_name(name_contains_averaged,
                                                      directory_path,
                                                      last_import_date_averaged_results
                                                      )
file_dict_summaries = download_files_with_name(name_contains_summary,
                                               directory_path,
                                               last_import_date_summaries)


def read_df_from_file_dict(file_dict):
    # Usage example
    file_data = read_json_files(file_dict.keys())

    for directory, data in file_data.items():
        modified_data = unwrap_graph_metadata(data)
        modified_data = add_object_storage_path(file_dict[directory], modified_data)
        modified_data = remove_redundant(modified_data)
        modified_data = rename_num_rounds(modified_data)
        file_data[directory] = modified_data

    file_data_values = list(file_data.values())

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(file_data_values)

    # Iterate over the columns
    for column in df.columns:
        try:
            if isinstance(df[column], dict):
                pass
            # Try converting the values to float
            df[column] = df[column].astype(float)
        except ValueError:
            # If conversion throws an exception, change the data type to string/text
            df[column] = df[column].astype(str)

    return df


df_averaged_results = read_df_from_file_dict(file_dict_averaged_results)
df_averaged_summaries = read_df_from_file_dict(file_dict_summaries)


# Save the DataFrame to the database
def insert_df_into_db_table(df, table_name):
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

        additional_columns = set(existing_columns) - set(df.columns)

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
        column_names = ', '.join(row.index)
        insert_query = f"INSERT OR IGNORE INTO {table_name} ({column_names}) VALUES ({', '.join(['?'] * len(row))});"
        cursor.execute(insert_query, tuple(row))


if df_averaged_results.size > 0:
    insert_df_into_db_table(df_averaged_results, table_name_averaged_results)
if df_averaged_summaries.size > 0:
    insert_df_into_db_table(df_averaged_summaries, table_name_summaries)

conn.commit()
# Close the database connection
conn.close()
