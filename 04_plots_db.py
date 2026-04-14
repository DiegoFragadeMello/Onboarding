import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.dates as mdates
from sqlalchemy import create_engine

DB_URI = "postgresql://sequor:sequor123@localhost:5432/sensor_demo"
ENGINE = create_engine(DB_URI)
DEFAULT_TAG = "T_Zone_01"


def load_tag_data(tag: str) -> pd.DataFrame:
    query = "SELECT * FROM sensor_readings WHERE tag = %(tag)s ORDER BY timestamp"
    df = pd.read_sql(query, ENGINE, params={"tag": tag}, parse_dates=["timestamp"])
    return df


def plot_dashboard(df: pd.DataFrame, tag: str) -> None:
    df_clean = df.dropna(subset=["value"]).reset_index(drop=True)

    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    fig.suptitle(f"Data Quality Dashboard - {tag}", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.axis("off")
    desc = df_clean["value"].describe().round(3).to_frame(name=tag)
    tbl = ax.table(
        cellText=desc.values,
        rowLabels=desc.index,
        colLabels=desc.columns,
        cellLoc="center",
        loc="center",
    )
    ax.set_title("Estatísticas descritivas")

    ax = axes[0, 1]
    ax.hist(df_clean["value"], bins=80, density=True, color="#3a6ea5", alpha=0.7)
    mu, sigma = df_clean["value"].mean(), df_clean["value"].std()
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "r--", lw=1.5)
    ax.axvline(mu + 3 * sigma, color="orange", ls="--", lw=1.5, label="+3σ")
    ax.axvline(mu - 3 * sigma, color="orange", ls="--", lw=1.5, label="-3σ")
    ax.set_title("Distribuição de valores")
    ax.legend()

    ax = axes[0, 2]
    ax.boxplot(
        [df_clean["value"]],
        tick_labels=[tag],
        patch_artist=True,
        boxprops=dict(facecolor="#3a6ea5", alpha=0.6),
        flierprops=dict(marker="x", markeredgecolor="#EE0000", markersize=5),
    )
    ax.set_title("Boxplot — outliers em vermelho")

    ax = axes[0, 3]
    ax.plot(df["timestamp"], df["value"], lw=0.5, alpha=0.7, color="steelblue")
    ax.set_title("Série temporal bruta")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))

    ax = axes[1, 0]
    roll = df_clean["value"].rolling(60)
    rm = roll.mean()
    rs = roll.std()
    ax.plot(
        df_clean["timestamp"], df_clean["value"], lw=0.4, alpha=0.5, color="steelblue"
    )
    ax.plot(df_clean["timestamp"], rm, color="navy", lw=1)
    ax.fill_between(
        df_clean["timestamp"], rm - 3 * rs, rm + 3 * rs, alpha=0.15, color="orange"
    )
    ax.set_title("Rolling mean ± 3σ")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))

    ax = axes[1, 1]
    is_outlier = (df_clean["value"] > mu + 3 * sigma) | (
        df_clean["value"] < mu - 3 * sigma
    )
    rate = is_outlier.rolling(200, min_periods=1).mean()
    ax.plot(df_clean["timestamp"], rate, color="crimson")
    ax.set_title("Taxa de anomalias (rolling)")
    ax.set_ylabel("Proporção")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))

    ax = axes[1, 2]
    utility_series = df_clean["value"].diff().abs()
    ax.plot(df_clean["timestamp"], utility_series, lw=0.4, color="indigo")
    ax.set_title("Variação absoluta ponto a ponto")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))

    ax = axes[1, 3]
    n_missing = df["value"].isna().astype(int).rolling(60, min_periods=1).sum()
    ax.plot(df["timestamp"], n_missing, lw=0.8, color="darkorange")
    ax.set_title("Contagem de valores ausentes (rolling 60)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))

    ax = axes[2, 0]
    grouped = (
        df["quality"]
        .value_counts(normalize=True)
        .reindex(["good", "bad"], fill_value=0)
    )
    grouped.plot(kind="bar", ax=ax, color=["#3a6ea5", "#ee7f0d"])
    ax.set_title("Qualidade dos dados")
    ax.set_ylabel("Proporção")

    ax = axes[2, 1]
    if "error_type" in df.columns:
        err_counts = df["error_type"].fillna("none").value_counts()
        err_counts.plot(kind="bar", ax=ax, color="#6a4b9f")
        ax.set_title("Tipos de erro")
        ax.set_ylabel("Contagem")
    else:
        ax.text(0.5, 0.5, "Sem coluna error_type", ha="center", va="center")
        ax.axis("off")

    for ax in [axes[2, 2], axes[2, 3]]:
        ax.axis("off")

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    plt.tight_layout()
    plt.savefig(f"dq_dashboard_db_{tag}.png", dpi=150, bbox_inches="tight")
    print(f"✓ Dashboard salvo em dq_dashboard_db_{tag}.png")


if __name__ == "__main__":
    tag = DEFAULT_TAG
    print(f"Carregando dados para {tag}...")
    df_tag = load_tag_data(tag)
    if df_tag.empty:
        raise ValueError(
            f"Nenhum dado encontrado para o tag '{tag}' no banco de dados."
        )

    plot_dashboard(df_tag, tag)
