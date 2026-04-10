import pandas as pd
import numpy as np
from scipy import stats

from statsmodels.tsa.stattools import adfuller

df = pd.read_parquet("sensor_anomalies.parquet")

# Completude
missing = df[["pressure", "temperature"]].isnull().sum() / len(df) * 100
print("Missing values percentage: ")
print(missing.round(2))

# Z-score

df_clean = df.dropna()
z_scores = np.abs(stats.zscore(df_clean[["pressure"]]))

outliers_z = df_clean[z_scores > 3]
print(f"Number of outliers detected by Z-score: {len(outliers_z)}")

# IQR

q1, q3 = df_clean["pressure"].quantile([0.25, 0.75])
iqr = q3 - q1
outliers_iqr = df_clean[
    (df_clean["pressure"] < q1 - 1.5 * iqr) | (df_clean["pressure"] > q3 + 1.5 * iqr)
]
print(f"Number of outliers detected by IQR: {len(outliers_iqr)}")

# ADF Test

adf_result = adfuller(df_clean["pressure"].values)
print(f"ADF p-value: {adf_result[1]:.4f}")
print(
    "Série ESTACIONARIA "
    if adf_result[1] < 0.05
    else "Drift detectado: série NÃO ESTACIONARIA"
)

# Gaps temporais

gaps = df["timestamp"].diff()
large_gaps = gaps[gaps > pd.Timedelta("2min")]
print(f"\n[GAPS] {len(large_gaps)} interrupções > 2 minutos.")

# Cosistencia

r, p = stats.pearsonr(df_clean["pressure"], df_clean["temperature"])
print(f"\n[CONSISTÊNCIA] r(P, T) = {r:.3f}, p-value = {p:.2e}")

# Sumario

report = {
    "completude": missing.to_dict(),
    "outliers_z_score": len(outliers_z),
    "outliers_iqr": len(outliers_iqr),
    "adf_p_value": round(adf_result[1], 4),
    "gaps_detectados": len(large_gaps),
    "correlacao_pearson": round(r, 3),
}
pd.Series(report).to_json("dq_report.json", indent=2)
print("\n Relatório de qualidade de dados salvo em 'dq_report.json'")
