import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.dates as mdates

df = pd.read_parquet("sensor_anomalies.parquet")
df_clean = df.dropna()

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle(
    "Data Quality Dashboard - Pressure & Temperature", fontsize=14, fontweight="bold"
)

ax = axes[1, 2]
ax.boxplot(
    [df_clean["temperature"]],
    tick_labels=["Temperatura (°C)"],
    patch_artist=True,
    boxprops=dict(facecolor="#3a6ea5", alpha=0.6),
    flierprops=dict(marker="x", markeredgecolor="#EE0000", markersize=5),
)
ax.set_title("Temperature boxplot — outliers visíveis em vermelho")


ax = axes[2, 0]
ax.hist(df_clean["pressure"], bins=80, density=True, color="steelblue", alpha=0.6)

mu, sigma = df_clean["pressure"].mean(), df_clean["pressure"].std()
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
ax.plot(x, stats.norm.pdf(x, mu, sigma), "r--", lw=1.5)
ax.axvline(mu + 3 * sigma, color="orange", ls="--", lw=1.5, label="+3σ")
ax.axvline(mu - 3 * sigma, color="orange", ls="--", lw=1.5, label="-3σ")
ax.set_title("Pressão — distribuição")
ax.legend()

ax = axes[2, 2]
ax.boxplot(
    [df_clean["pressure"]],
    tick_labels=["Pressão (bar)"],
    patch_artist=True,
    boxprops=dict(facecolor="#3a6ea5", alpha=0.6),
    flierprops=dict(marker="x", markeredgecolor="#EE0000", markersize=5),
)
ax.set_title("Pressão boxplot — outliers visíveis em vermelho")

ax = axes[0, 1]
sc = ax.scatter(
    df_clean["pressure"],
    df_clean["temperature"],
    s=1,
    alpha=0.3,
    c=df_clean.index,
    cmap="viridis",
)
ax.set_xlabel("Pressão (bar)")
ax.set_ylabel("Temperatura (°C)")
ax.set_title("Pressão × Temperatura (correlação)")

ax = axes[1, 3]
roll = df_clean["pressure"].rolling(60)
rm = roll.mean()
rs = roll.std()
ax.plot(df_clean.index, df_clean["pressure"], lw=0.3, alpha=0.5, color="steelblue")
ax.plot(df_clean.index, rm, color="navy", lw=1)
ax.fill_between(df_clean.index, rm - 3 * rs, rm + 3 * rs, alpha=0.15, color="orange")
ax.set_title("Rolling mean ± 3σ (Detecção de drift)")

ax = axes[0, 0]
ax.axis("off")
desc = df_clean[["pressure", "temperature"]].describe().round(3)
tbl = ax.table(
    cellText=desc.values,
    rowLabels=desc.index,
    colLabels=desc.columns,
    cellLoc="center",
    loc="center",
)
ax = axes[1, 0]

ax.hist(df_clean["temperature"], bins=80, density=True, color="steelblue", alpha=0.6)
mu, sigma = df_clean["temperature"].mean(), df_clean["temperature"].std()
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
ax.plot(x, stats.norm.pdf(x, mu, sigma), "r--", lw=1.5)
ax.axvline(mu + 3 * sigma, color="orange", ls="--", lw=1.5, label="+3σ")
ax.axvline(mu - 3 * sigma, color="orange", ls="--", lw=1.5, label="-3σ")
ax.set_title("Temperatura — distribuição")
ax.legend()

ax = axes[2, 3]
ax.plot(df["timestamp"], df["pressure"], lw=0.4, alpha=0.7, color="steelblue")
ax.set_title("Pressão — série bruta + drift")
ax.set_ylabel("bar")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))

ax = axes[0, 2]

mu = df_clean["pressure"].mean()
sigma = df_clean["pressure"].std()

is_outlier = (df_clean["pressure"] > mu + 3 * sigma) | (
    df_clean["pressure"] < mu - 3 * sigma
)

rate = is_outlier.rolling(200).mean()

ax.plot(rate, color="crimson")
ax.set_title("Taxa de anomalias (rolling)")
ax.set_ylabel("Proporção")

ax = axes[2, 1]
ax.plot(df["timestamp"], df["temperature"], lw=0.4, alpha=0.7, color="steelblue")
ax.set_title("Temperatura — série bruta + drift")
ax.set_ylabel("°C")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))

ax = axes[1, 1]
roll = df_clean["temperature"].rolling(60)
rm = roll.mean()
rs = roll.std()
ax.plot(df_clean.index, df_clean["temperature"], lw=0.3, alpha=0.5, color="steelblue")
ax.plot(df_clean.index, rm, color="navy", lw=1)
ax.fill_between(df_clean.index, rm - 3 * rs, rm + 3 * rs, alpha=0.15, color="orange")
ax.set_title("Rolling mean ± 3σ (Detecção de drift)")

ax = axes[0, 3]

ax.plot(
    df_clean["timestamp"],
    df_clean["temperature"],
    label="Temperatura",
    color="steelblue",
)
ax.plot(
    df_clean["timestamp"],
    df_clean["pressure"],
    label="Pressão",
    color="navy",
)


ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))

ax.legend()
ax.set_title("Pressão e Temperatura ao longo do tempo")

tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
ax.set_title("Estatísticas descritivas")
plt.tight_layout()
plt.savefig("dq_dashboard.png", dpi=150, bbox_inches="tight")
print("✓ Dashboard salvo em dq_dashboard.png")
