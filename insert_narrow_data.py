import pandas as pd
from sqlalchemy import create_engine

# Conexão com o banco de dados
ENGINE = create_engine("postgresql://sequor:sequor123@localhost:5432/sensor_demo")

# Ler o CSV narrow
df = pd.read_csv("furnace_v2_narrow.csv", parse_dates=["timestamp"])

# Inserir no banco de dados
df.to_sql("sensor_readings", ENGINE, if_exists="append", index=False, method="multi")

print(f"Inserido {len(df)} linhas na tabela sensor_readings.")
