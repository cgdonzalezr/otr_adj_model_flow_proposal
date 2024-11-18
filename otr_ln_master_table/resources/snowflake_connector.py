import snowflake.connector
import pandas as pd


def snowflake_conn(database: str = None, schema: str = None, role: str = 'DATA_SCIENCE_ROLE', user: str = 'USER') -> snowflake.connector.SnowflakeConnection:
  """
  Establishes a connection to a Snowflake database.

  Args:
      database (str, optional): The name of the Snowflake database to connect to. Defaults to None.
      schema (str, optional): The name of the Snowflake schema to connect to. Defaults to None.
      role (str, optional): The Snowflake role to use for the connection. Defaults to 'DATA_SCIENCE_ROLE'.
      user (str, optional): The Snowflake username to use for the connection. Defaults to 'USER' (place your actual username here).

  Returns:
      snowflake.connector.SnowflakeConnection: A Snowflake connector object representing the connection.

  Raises:
      ValueError: If the provided user parameter is 'USER'. You must replace 'USER' with your actual Snowflake username.
  """

  if user == 'USER':
      raise ValueError("Please replace 'USER' with your actual Snowflake username.")

  con_eb = snowflake.connector.connect(
      user=user,
      account='bqa07840.us-west-2.privatelink',
      authenticator="externalbrowser",
      database=database,
      schema=schema,
      role=role,
      autocommit=True
  )
  return con_eb






def sf_get_data(query: str, snowflake_connection: snowflake.connector.SnowflakeConnection) -> pd.DataFrame:
  """
  Executes a SQL query on a Snowflake connection and returns the results as a Pandas DataFrame.

  Args:
      query (str): The SQL query to execute on the Snowflake database.
      snowflake_connection (snowflake.connector.SnowflakeConnection): A Snowflake connector object representing an open connection to the database.

  Returns:
      pd.DataFrame: A Pandas DataFrame containing the results of the executed query.

  Prints:
      The shape of the resulting DataFrame and the execution time of the query in minutes.
  """

  conn = snowflake_connection
  cur = conn.cursor()

  cur.execute(query)
  retrain_df = pd.DataFrame.from_records(iter(cur), columns=[x[0] for x in cur.description])

  return retrain_df

