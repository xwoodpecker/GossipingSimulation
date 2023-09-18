import sqlite3
import os

# Get the script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))+"\\in"

# List of input database filenames with a specific extension (e.g., .db)
input_db_files = [file for file in os.listdir(script_directory) if file.endswith('.db')]

# Name of the output merged database file
output_db_file = 'gnutella-series.db'

# Connect to the output database (or create it if it doesn't exist)
output_conn = sqlite3.connect(output_db_file)
output_cursor = output_conn.cursor()

# Tables you want to merge
tables_to_merge = ['SUMMARIES', 'AVERAGED_RESULTS']

# Create the tables in the output database with the same structure as the input tables
for table_name in tables_to_merge:
    for input_db_file in input_db_files:
        # Connect to the input database
        input_db_path = os.path.join(script_directory, input_db_file)
        input_conn = sqlite3.connect(input_db_path)
        input_cursor = input_conn.cursor()

        try:
            # Check if the table exists in the input database
            input_cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
            input_table_exists = input_cursor.fetchone()

            if input_table_exists:
                # Retrieve the table schema from the input database
                input_cursor.execute(f"PRAGMA table_info({table_name});")
                input_columns = input_cursor.fetchall()
                input_column_names = [col[1] for col in input_columns]
                input_column_types = [col[2] for col in input_columns]

                output_cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
                output_table_exists = output_cursor.fetchone()

                if not output_table_exists:
                    # Generate the CREATE TABLE statement for the output database
                    create_table_query = \
                        f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join([f'{name} {type}' for name, type in zip(input_column_names, input_column_types)])});"

                    # Execute the CREATE TABLE statement in the output database
                    output_cursor.execute(create_table_query)
                    print(f"Table {table_name} created in {output_db_file}")

                else:
                    # Retrieve the table schema from the output database (if it exists)
                    output_cursor.execute(f"PRAGMA table_info({table_name});")
                    output_columns = output_cursor.fetchall()
                    output_column_names = [col[1] for col in output_columns]

                    # Calculate the columns that are in the input but not in the output
                    new_columns = [col for col in input_column_names if col not in output_column_names]

                    if new_columns:
                        # Generate ALTER TABLE statements to add new columns
                        alter_table_queries = [f"ALTER TABLE {table_name} ADD COLUMN {col};" for col in new_columns]

                        # Execute the ALTER TABLE statements in the output database
                        for alter_query in alter_table_queries:
                            output_cursor.execute(alter_query)
                            print(f"Table {table_name} altered in {output_db_file} - Added columns: {', '.join(new_columns)}")

        except sqlite3.Error as e:
            print(f"Error creating or altering table {table_name} in {output_db_file}: {e}")

# Iterate through input database files
for input_db_file in input_db_files:
    # Connect to the input database
    input_db_path = os.path.join(script_directory, input_db_file)
    input_conn = sqlite3.connect(input_db_path)
    input_cursor = input_conn.cursor()

    # Iterate through tables to merge
    for table_name in tables_to_merge:
        try:
            # Check if the table exists in the input database
            input_cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
            table_exists = input_cursor.fetchone()
            if table_exists:
                # Select all data from the current table in the input database
                input_cursor.execute(f"SELECT * FROM {table_name};")
                data = input_cursor.fetchall()

                # Extract column names from the cursor description
                column_names = [col[0] for col in input_cursor.description]

                # Generate the INSERT INTO statement with explicit column names
                insert_query = f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES ({', '.join(['?'] * len(column_names))});"

                # Insert the data into the output table
                output_cursor.executemany(insert_query, data)
                print(f"Inserted {len(data)} rows into {table_name} in {output_db_file}")
        except sqlite3.Error as e:
            print(f"Error inserting data into table {table_name} in {input_db_file}: {e}")

# Commit changes for the output database and close the connection
output_conn.commit()
output_conn.close()

print("Database merge completed.")
