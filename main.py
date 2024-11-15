import os

import joblib
import psycopg2
import select
import pandas as pd
import numpy as np
import csv
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
import ray
from ray.util.actor_pool import ActorPool
from ray.util.queue import Queue
from table_insert import start_table_insert


@ray.remote
class Monitor:
    def __init__(self, workers_pool):
        self.workers_pool = workers_pool
        self.queue = Queue()

    def push(self, item):
        self.queue.put(item)
        if self.workers_pool.has_free():
            self.send_task()
        else:
            self.workers_pool.get_next()
            self.send_task()


    def send_task(self):
        self.workers_pool.submit(lambda a, v: a.work.remote(v), self.queue.get())


    def monitor(self):
        while not self.queue.empty():
            print(self.workers_pool.has_free())
            print(self.queue.qsize())
            if self.workers_pool.has_free():
                self.workers_pool.submit(lambda a, v: a.work.remote(v), self.queue.get())

            if self.workers_pool.has_next():
                self.workers_pool.get_next()
                self.workers_pool.submit(lambda a, v: a.work.remote(v), self.queue.get())


# Predictor class
@ray.remote
class Predictor:
    def __init__(self, model_path):
        tf.config.experimental.set_visible_devices([], "GPU")
        self.model = load_model(model_path)

    def predict(self, data):
        prediction = self.model.predict(data, verbose=0)
        return prediction[0][0]

'''
@ray.remote
def worker(db_config, row_id, predictors_pool, scaler_path, lambdas_wealth_income, column_names):
    conn = psycopg2.connect(**db_config)
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    scaler = joblib.load(scaler_path)

    fitted_lambda_wealth, fitted_lambda_income = lambdas_wealth_income

    print(f"Worker {os.getpid()} is processing the row with ID={row_id}...")

    # Fetch and process the row based on ID
    print(f"Worker is fetching the row with ID={row_id}...")
    query = f"""SELECT * 
                                FROM needs 
                                WHERE ID = {row_id}"""
    cursor.execute(query)
    tuples_list = cursor.fetchall()
    data = pd.DataFrame(tuples_list, columns=column_names)
    print(f"Worker {os.getpid()} fetched the row with ID={row_id}:")

    # Data prep
    data['FinancialStatus'] = data['FinancialEducation'] * np.log(data['Wealth'])
    query = f"""UPDATE needs
                                SET financialstatus = {data['FinancialStatus'].iloc[0]}
                                WHERE id = {row_id}"""
    cursor.execute(query)
    print(
        f"Worker {os.getpid()} computed and updated for ID={row_id} the FinancialStatus: {data['FinancialStatus'].iloc[0]}")

    data = data.drop('ID', axis=1)
    data = data.drop('RiskPropensity', axis=1)

    data['Wealth'] = (data['Wealth'] ** fitted_lambda_wealth - 1) / fitted_lambda_wealth
    data['Income'] = (data['Income'] ** fitted_lambda_income - 1) / fitted_lambda_income

    data_scaled = scaler.transform(data)
    print(f"Worker {os.getpid()} preprocessed the row with ID={row_id}:")

    # Predict
    predictors_pool.submit(lambda a, v: a.predict.remote(v[0], v[1]), (data_scaled, row_id))
    prediction = predictors_pool.get_next()
    print(f"Worker {os.getpid()} predicted RiskPropensity for ID={row_id}: {prediction}")

    # Update DB with prediction result
    query = f"""UPDATE needs 
                                SET riskpropensity = {prediction}
                                WHERE id = {row_id}"""
    cursor.execute(query)
    print(f"Worker {os.getpid()} updated for ID={row_id} the RiskPropensity: {prediction}")

    # Get best products
    query = f"""SELECT * FROM products
                                WHERE (Income  = {data['IncomeInvestment'].iloc[0]} 
                                        OR Accumulation = {data['AccumulationInvestment'].iloc[0]})
                                AND Risk <= {prediction}"""
    cursor.execute(query)
    tuples_list = cursor.fetchall()
    print(f"Worker {os.getpid()} advises {len(tuples_list)} products for ID={row_id}:")
    # Print the results
    for row in tuples_list:
        print(row)

    return
'''

@ray.remote
class Worker:
    def __init__(self, db_config, predictors_pool, scaler_path, lambdas_wealth_income, column_names, log_file_path):
        self.db_config = db_config
        self.conn = psycopg2.connect(**db_config)
        self.conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        self.cursor = self.conn.cursor()
        self.predictors_pool = predictors_pool
        self.scaler = joblib.load(scaler_path)
        self.fitted_lambda_wealth, self.fitted_lambda_income = lambdas_wealth_income
        self.column_names = column_names
        self.log_file_path = log_file_path

    def log_activity(self, row_id):
        """
        Logs the worker's PID and the processed row ID to a CSV file.
        """
        with open(self.log_file_path, mode='a', newline='') as log_file:
            writer = csv.writer(log_file)
            writer.writerow([datetime.now().isoformat(), os.getpid(), row_id])

    def work(self, row_id):
        self.log_activity(row_id)
        print(f"Worker {os.getpid()} is processing the row with ID={row_id}...")

        # Fetch and process the row based on ID
        print(f"Worker is fetching the row with ID={row_id}...")
        query = f"""SELECT * 
                            FROM needs 
                            WHERE ID = {row_id}"""
        self.cursor.execute(query)
        tuples_list = self.cursor.fetchall()
        data = pd.DataFrame(tuples_list, columns=self.column_names)
        print(f"Worker {os.getpid()} fetched the row with ID={row_id}:")

        # Data prep
        data['FinancialStatus'] = data['FinancialEducation'] * np.log(data['Wealth'])
        query = f"""UPDATE needs
                            SET financialstatus = {data['FinancialStatus'].iloc[0]}
                            WHERE id = {row_id}"""
        self.cursor.execute(query)
        print(
            f"Worker {os.getpid()} computed and updated for ID={row_id} the FinancialStatus: {data['FinancialStatus'].iloc[0]}")

        data = data.drop('ID', axis=1)
        data = data.drop('RiskPropensity', axis=1)

        data['Wealth'] = (data['Wealth'] ** self.fitted_lambda_wealth - 1) / self.fitted_lambda_wealth
        data['Income'] = (data['Income'] ** self.fitted_lambda_income - 1) / self.fitted_lambda_income

        data_scaled = self.scaler.transform(data)
        print(f"Worker {os.getpid()} preprocessed the row with ID={row_id}:")

        # Predict
        self.predictors_pool.submit(lambda a, v: a.predict.remote(v), data_scaled)
        prediction = self.predictors_pool.get_next()
        print(f"Worker {os.getpid()} predicted RiskPropensity for ID={row_id}: {prediction}")

        # Update DB with prediction result
        query = f"""UPDATE needs 
                            SET riskpropensity = {prediction}
                            WHERE id = {row_id}"""
        self.cursor.execute(query)
        print(f"Worker {os.getpid()} updated for ID={row_id} the RiskPropensity: {prediction}")

        # Get best products
        query = f"""SELECT * FROM products
                            WHERE (Income  = {data['IncomeInvestment'].iloc[0]} 
                                    OR Accumulation = {data['AccumulationInvestment'].iloc[0]})
                            AND Risk <= {prediction}"""
        self.cursor.execute(query)
        tuples_list = self.cursor.fetchall()
        print(f"Worker {os.getpid()} advises {len(tuples_list)} products for ID={row_id}:")
        # Print the results
        for row in tuples_list:
            print(row)

        return


if __name__ == '__main__':
    db_config = {
        "dbname": "postgres",
        "user": "postgres",
        "password": "tony",
        "host": "localhost",
        "port": "5432"
    }

    column_names = [
        "ID", "Age", "Gender", "FamilyMembers", "FinancialEducation",
        "RiskPropensity", "Income", "Wealth", "IncomeInvestment",
        "AccumulationInvestment", "FinancialStatus"
    ]

    model_path = 'deep_model.h5'
    scaler_path = 'deep_scaler.pkl'
    lambdas_wealth_income = [0.1336735055366279, 0.3026418664067109]

    # Initialize Ray
    ray.init()

    # Queue def
    #queue = Queue()

    # Predictors pool
    num_predictors = 2
    predictors = [Predictor.remote(model_path) for _ in range(num_predictors)]
    predictors_pool = ActorPool(predictors)


    # Workers pool
    num_workers = 7

    log_file_path = 'worker_logs.csv'

    # Ensure the log file has a header row
    with open(log_file_path, mode='w', newline='') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["Timestamp", "Worker_PID", "Row_ID"])

    # Workers pool
    workers = [
        Worker.remote(db_config, predictors_pool, scaler_path, lambdas_wealth_income, column_names, log_file_path)
        for _ in range(num_workers)]

    #workers = [Worker.remote(db_config, predictors_pool, scaler_path, lambdas_wealth_income, column_names) for _ in range(num_workers)]
    workers_pool = ActorPool(workers)

    # Monitor init
    monitor = Monitor.remote(workers_pool)
    #monitor.monitor.remote()

    # DB config
    conn = psycopg2.connect(**db_config)
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    # Listen on table_insert channel
    cursor.execute("LISTEN table_insert;")
    print("Main listener is active and waiting for events...")
    query = "delete from needs where id > 5000"
    cursor.execute(query)
    start_table_insert(1000)

    try:
        #counter = 0
        while True:
            # Wait for events
            if select.select([conn], [], [], 5) != ([], [], []):
                conn.poll()

                while conn.notifies:
                    notify = conn.notifies.pop(0)
                    row_id = int(notify.payload)
                    print(f"Notification received for ID={row_id}")
                    monitor.push.remote(row_id)

            '''        
            print(workers_pool.has_next())
            print(queue.qsize())
            if workers_pool.has_next() and not queue.empty():
                workers_pool.get_next()
                workers_pool.submit(lambda a, v: a.work.remote(v), queue.get())
            '''

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        cursor.close()
        conn.close()