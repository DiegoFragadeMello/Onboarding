import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta

ENGINE = create_engine("postgresql://sequor:sequor123@localhost:5432/sensor_demo")

rng = pd.date_range(start="2026-04-08", periods=10080, freq="1min")
np.random.seed(42)

df = pd.DataFrame(
    {
        "time": rng,
        "sensor_id": "SENSOR-A",
        "pressure": 5.0 + np.random.uniform(0, 0.1, len(rng)),
        "temperature": 70.0 + np.random.uniform(0, 0.5, len(rng)),
    }
)
df.to_sql("sensor_readings", ENGINE, if_exists="append", index=False, method="multi")
print(f"Inserted {len(df)} rows into the database.")
