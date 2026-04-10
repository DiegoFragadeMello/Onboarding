import numpy as np
import pandas as pd

np.random.seed(0)

N = 10_080
t = np.arange(N)

# Ruido gaussiano normal
pressure = 5.0 + np.random.normal(0, 0.08, N)
temperature = (
    70.0 + 0.3 * (pressure - 5.0) + np.random.normal(0, 0.4, N)
)  # <-- Corelacao entre temperatura e pressao

#   Cenario de falha numero 01
#       Drift

drift_mask = t > 5000
pressure[drift_mask] += (t[drift_mask] - 5000) * 0.0001

#   Cenario de falha numero 02
#       Outliers

outlier_idx = np.random.choice(N, 12, replace=False)
pressure[outlier_idx] += np.random.uniform(1.5, 3.0, 12)
temperature[outlier_idx] += np.random.uniform(8.0, 15.0, 12)

#   Cenario de falha numero 03
#       Failure

pressure[7200:7245] = np.nan
temperature[7200:7245] = np.nan

rng = pd.date_range(start="2024-01-01", periods=N, freq="1min")
df = pd.DataFrame({"timestamp": rng, "pressure": pressure, "temperature": temperature})

df.to_parquet(
    "sensor_anomalies.parquet",
)
print("Dataset 'sensor_anomalies.parquet' criado com sucesso!")
