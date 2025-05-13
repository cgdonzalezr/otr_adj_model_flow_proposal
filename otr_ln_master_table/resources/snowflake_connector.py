import os
import numpy as np
import snowflake.connector
import pandas as pd

# Environment Variables
from dotenv import load_dotenv
load_dotenv()

user = os.getenv("user")
account = os.getenv("account")
warehouse = os.getenv("warehouse")
database = os.getenv("database")
schema = os.getenv("schema")
authenticator = os.getenv("authenticator")

# Connect to Snowflake
conn = snowflake.connector.connect(
    user=user,
    account=account,
    warehouse=warehouse,
    database=database,  
    schema=schema,
    authenticator=authenticator
)
