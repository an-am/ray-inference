import psycopg2
import random

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

def start_table_insert(count):
    for _ in range(count):
        age = random.randint(18, 90)
        gender = random.randint(0, 1)
        family_members = random.randint(0, 5)
        financial_education = random.uniform(0.036098897, 0.902932641)
        income = random.uniform(1.537764666, 365.3233855)
        wealth = random.uniform(1.057414979, 2233.228433)
        income_investment = random.randint(0, 1)
        accumulation_investment = random.randint(0, 1)
        client_id = random.randint(1, 100)

        query = f"""INSERT INTO Needs (ID, Age, Gender, FamilyMembers, FinancialEducation, 
                                        Income, Wealth, IncomeInvestment, AccumulationInvestment, ClientId)
                    VALUES (
                    (SELECT MAX(ID)+1 FROM Needs),
                    {age}, {gender}, {family_members}, {financial_education},
                    {income}, {wealth}, {income_investment}, {accumulation_investment},
                    {client_id}
                );"""

        cursor.execute(query)

    cursor.close()
    conn.close()
