import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_parquet("sensor_anomalies.parquet")
df = df.dropna()
fig, axes = plt.subplots(5, 1, figsize=(14, 12))
fig.suptitle("Dashboard - Séries Temporais (linha)", fontsize=16, fontweight="bold")
axes[0].plot(df["timestamp"], df["pressure"], color="steelblue", label="Pressão (bar)")
axes[0].set_title("Pressão ao longo do tempo")
axes[0].set_ylabel("bar")
axes[0].legend()

axes[1].plot(
    df["timestamp"], df["temperature"], color="green", label="Temperatura (°C)"
)
axes[1].set_title("Temperatura ao longo do tempo")
axes[1].set_ylabel("°C")
axes[1].legend()

roll_p = df["pressure"].rolling(60)

axes[2].plot(df["timestamp"], df["pressure"], alpha=0.4)

axes[2].plot(df["timestamp"], roll_p.mean(), color="navy")

axes[2].set_title("Pressão + Rolling Mean")

# 4 — Temperatura + média móvel

roll_t = df["temperature"].rolling(60)

axes[3].plot(df["timestamp"], df["temperature"], alpha=0.4)

axes[3].plot(df["timestamp"], roll_t.mean(), color="darkred")

axes[3].set_title("Temperatura + Rolling Mean")

# 5 — Taxa de anomalias

mu = df["pressure"].mean()
sigma = df["pressure"].std()

outlier = (df["pressure"] > mu + 3 * sigma) | (df["pressure"] < mu - 3 * sigma)

rate = outlier.rolling(200).mean()

axes[4].plot(df["timestamp"], rate, color="black")

axes[4].set_title("Taxa de Anomalias")

# Formatar datas

for ax in axes:
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))

plt.xticks(rotation=45)


plt.tight_layout()
plt.savefig("dq_dashboardSerieLinha.png", dpi=150, bbox_inches="tight")
print("✓ Dashboard salvo em dq_dashboardSerieLinha.png")
