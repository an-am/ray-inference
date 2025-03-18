import psycopg2
import random
from pandas import DataFrame


def copy_first_1000_rows():
    db_config = {
        "dbname": "postgres",
        "user": "postgres",
        "password": "tony",
        "host": "localhost",
        "port": "5432"
    }

    # Establish connection
    conn = psycopg2.connect(**db_config)
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    # Step 1: Fetch first 1000 rows
    cursor.execute("SELECT * FROM Needs ORDER BY ID ASC LIMIT 1000")
    rows = cursor.fetchall()

    # Step 2: Extract column names dynamically
    cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'needs'")
    columns = [col[0] for col in cursor.fetchall()]

    # Find the index of the "ID" column
    id_index = columns.index("id")

    # Step 3: Modify the ID values (start from 5001)
    new_rows = []
    for i, row in enumerate(rows):
        row = list(row)  # Convert tuple to list to modify ID
        row[id_index] = 5001 + i  # Set new ID
        new_rows.append(tuple(row))

    # Step 4: Prepare the INSERT query
    insert_query = f"INSERT INTO Needs ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))})"

    # Step 5: Execute batch insert
    cursor.executemany(insert_query, new_rows)
    print(f"Inserted {len(new_rows)} new rows starting from ID 5001.")

    # Close connections
    cursor.close()
    conn.close()


# Run the function
copy_first_1000_rows()