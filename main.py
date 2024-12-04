import joblib
import psycopg2
import select
import requests
import pandas as pd
import numpy as np
import ray
from ray.util.queue import Queue
from table_insert import start_table_insert


@ray.remote(num_cpus=1, resources={"monitor_node":1})
class Monitor:
    def __init__(self, num_workers, workers):
        self.task_queue = Queue()
        self.worker_ids = [i for i in range(num_workers)]
        self.workers = workers
        self.record = pd.DataFrame(columns=['Worker', 'ClientId', 'Count'])
        self.allowed_requests = 5

    def push(self, row_id, client_id):
        self.task_queue.put(row_id)
        self.check_record(client_id)

    def check_record(self, client_id):
        # Assign client to worker
        assigned_worker_id = client_id % len(self.worker_ids)

        # Check if the row exists
        existing_row = self.record[self.record['ClientId'] == client_id]

        if existing_row.empty:
            # Row doesn't exist, add a new one with Count = 1
            new_row = {'Worker': assigned_worker_id, 'ClientId': client_id, 'Count': 1}
            self.record = pd.concat([self.record, pd.DataFrame([new_row])], ignore_index=True)
            self.workers[assigned_worker_id].work.remote(self.task_queue.get())

        else:
            # Row exists, increment the Count
            if self.record.at[existing_row.index[0], 'Count'] < self.allowed_requests:
                index = existing_row.index[0]  # Index of the row
                self.record.at[index, 'Count'] += 1
                self.workers[assigned_worker_id].work.remote(self.task_queue.get())

            else:
                self.task_queue.get()
                #print(f"Client {client_id} has exceeded allowed requests")


@ray.remote(num_cpus=1, resources={"workers_node":1})
class Worker:
    def __init__(self, db_config, scaler_path, lambdas_wealth_income, column_names):
        self.db_config = db_config
        self.conn = psycopg2.connect(**db_config)
        self.conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        self.cursor = self.conn.cursor()
        self.scaler = joblib.load(scaler_path)
        self.fitted_lambda_wealth, self.fitted_lambda_income = lambdas_wealth_income
        self.column_names = column_names

    def fetch_row(self, row_id):
        '''
        Fetch row from DB and returns it into a df
        '''
        query = f"""SELECT * 
                    FROM needs 
                    WHERE ID = {row_id}"""

        self.cursor.execute(query)
        tuples_list = self.cursor.fetchall()

        return pd.DataFrame(tuples_list, columns=self.column_names)

    def data_prep(self, data):
        data['FinancialStatus'] = data['FinancialEducation'] * np.log(data['Wealth'])
        financial_status = data['FinancialStatus'].iloc[0]

        data = data.drop('ID', axis=1)
        data = data.drop('RiskPropensity', axis=1)
        data = data.drop('ClientId', axis=1)

        data['Wealth'] = (data['Wealth'] ** self.fitted_lambda_wealth - 1) / self.fitted_lambda_wealth
        data['Income'] = (data['Income'] ** self.fitted_lambda_income - 1) / self.fitted_lambda_income

        data_scaled = self.scaler.transform(data)

        return data_scaled, financial_status

    def work(self, row_id):
        print(f"Processing the row with ID={row_id}...")

        # Get the tuple from DB
        data = self.fetch_row(row_id)
        print(f"Fetched the row with ID={row_id}")

        # Data prep
        data_scaled, financial_status = self.data_prep(data)
        print(f"Preprocessed the row with ID={row_id}")

        # Prediction
        response = requests.get(
            "http://localhost:8000/", json={"array": data_scaled.tolist()}
        )
        prediction = response.json()
        prediction = prediction["prediction"]
        print(f"Predicted RiskPropensity for ID={row_id}: {prediction}")

        # Update DB: FinancialStatus and RiskPropensity
        query = f"""UPDATE needs 
                    SET riskpropensity  = {prediction},
                        financialstatus = {financial_status}
                    WHERE id = {row_id}"""
        self.cursor.execute(query)
        print(f"Updated for ID={row_id} RiskPropensity={prediction} and FinancialStatus={financial_status}")

        # Get best products
        query = f"""SELECT * 
                    FROM products
                    WHERE   
                        (
                            Income  = {data['IncomeInvestment'].iloc[0]} 
                            OR Accumulation = {data['AccumulationInvestment'].iloc[0]}
                        )
                        AND Risk <= {prediction}"""
        self.cursor.execute(query)
        tuples_list = self.cursor.fetchall()

        # Print the results
        print(f"Advised {len(tuples_list)} products for ID={row_id}")
        #for row in tuples_list:
        #    print(row)


if __name__ == '__main__':
    db_config = {
        "dbname": "postgres",
        "user": "postgres",
        "password": "tony",
        "host": "localhost",
        "port": "5432"
    }

    model_path = 'deep_model.h5'
    scaler_path = 'deep_scaler.pkl'
    lambdas_wealth_income = [0.1336735055366279, 0.3026418664067109]
    column_names = [
        "ID", "Age", "Gender", "FamilyMembers", "FinancialEducation",
        "RiskPropensity", "Income", "Wealth", "IncomeInvestment",
        "AccumulationInvestment", "FinancialStatus", "ClientId"
    ]

    # Initialize Ray
    ray.init()

    # Workers pool
    num_workers = 6

    # Workers pool
    workers = [Worker.remote(db_config, scaler_path,
                             lambdas_wealth_income, column_names)
               for _ in range(num_workers)]

    # Monitor init
    monitor = Monitor.remote(num_workers, workers)

    # DB config
    conn = psycopg2.connect(**db_config)
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    # Listen on table_insert channel
    cursor.execute("LISTEN table_insert;")
    print("Main listener is active and waiting for events...")
    ### the following block is for testing ###
    query = "delete from needs where id > 5000"
    cursor.execute(query)
    start_table_insert(1000)
    ### the previous block is for testing ###
    try:
        while True:
            # Wait for events
            if select.select([conn], [], [], 5) != ([], [], []):
                conn.poll()

                while conn.notifies:
                    notify = conn.notifies.pop(0)
                    data = dict(item.split('=') for item in notify.payload.split(','))
                    row_id = int(data['ID'])
                    client_id = int(data['ClientID'])
                    print(f"Received Notification: ID={row_id}, ClientID={client_id}")
                    monitor.push.remote(row_id, client_id)

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        cursor.close()
        conn.close()