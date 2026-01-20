# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 20:46:04 2025

@author: DELL
"""
###
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 22:31:37 2025

@author: DELL
"""
import os
from sqlalchemy import create_engine, inspect, text
import pandas as pd


from configparser import ConfigParser
from sqlalchemy import create_engine

def load_db_config(path = "C:/Users/DELL/kaggle_patient_journey/mysqlconfig.ini", section="database"):
    parser = ConfigParser()
    parser.read(path)
    return parser[section]

cfg = load_db_config()

user     = cfg["user"]
password = cfg["password"]
host     = cfg["host"]
port     = cfg["port"]
dbname   = cfg["dbname"]

# Build a MySQL connection string
connection_string = f"mysql+pymysql://{user}:{password}@{host}/{dbname}"

# Create the SQLAlchemy engine
engine = create_engine(connection_string)

# (Optional) Test connection
with engine.connect() as conn:
    print("Connected successfully!")


# Query the table and load it into a DataFrame

# Create an inspector to get table names
inspector = inspect(engine)
tables = inspector.get_table_names()
views = inspector.get_view_names()
# Dictionary to store DataFrames
dfs = {}

# Loop through all tables and read them into DataFrames
with engine.connect() as conn:
    for table in tables:
        dfs[table] = pd.read_sql(text(f"SELECT * FROM {table}"), con=conn)
        print(f"Loaded table '{table}' with {len(dfs[table])} rows.")
    for view in views:
        dfs[view] = pd.read_sql(text(f"SELECT * FROM {view}"), con=conn)
        print(f"Loaded view '{view}' with {len(dfs[view])} rows.")
    
# Example: access a specific table's DataFrame
for table_name, df in dfs.items():
    globals()[f"{table_name}_df"] = df
    print(f"Created variable: {table_name}_df")



### Handle dataframes
#encounters with admission type = outpatient, emergency -> length of stay is 0
import numpy as np

encounters_df['length_of_stay'] = np.where(
    (encounters_df['length_of_stay'].isna()) & (encounters_df['visit_type'] == 'Outpatient'),
    0,
    encounters_df['length_of_stay']
)

encounters_df['length_of_stay'] = np.where(
    (encounters_df['length_of_stay'].isna()) & (encounters_df['visit_type'] == 'Emergency'),
    0,
    encounters_df['length_of_stay']
)


encounters_plot = encounters_df[['visit_type',
       'department',
       'length_of_stay', 'readmitted_flag']]
#remove rows where both visit status types and Length of stays are null
encounters_plot = encounters_plot.dropna(subset=['length_of_stay'])

cat_cols = ["visit_type",
       "department", "readmitted_flag"]
# For each categorical column, generate dummies and join to original df
for col in cat_cols:
    dummies = pd.get_dummies(encounters_plot[col], prefix=col, dtype=int)
    encounters_plot = pd.concat([encounters_plot, dummies], axis=1)

# Optionally drop the original categorical columns
encounters_plot = encounters_plot.drop(columns=cat_cols)

## Distribution scatterplot for variables in encounters
import seaborn as sns
import matplotlib.pyplot as plt

#When to Use Spearman Correlation:

# The variables are not normally distributed or don't have a linear relationship.
# The variables are ordinal or ranked rather than continuous.
# There's evidence of a monotonic relationship but not necessarily a linear one.

encounters_plot_spearman = encounters_plot.corr(method='spearman')

plt.figure(figsize=(10, 8))
sns.heatmap(encounters_plot_spearman, cmap="coolwarm", center=0)
plt.title("Correlation Matrix (Before PCA)")
plt.show()

