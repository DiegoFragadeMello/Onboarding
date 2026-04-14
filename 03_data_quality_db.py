import json
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sqlalchemy import create_engine

DB_URI = "postgresql://sequor:sequor123@localhost:5432/sensor_demo"
ENGINE = create_engine(DB_URI)


def load_sensor_data() -> pd.DataFrame:
    query = "SELECT * FROM sensor_readings ORDER BY timestamp"
    df = pd.read_sql(query, ENGINE, parse_dates=["timestamp"])
    return df


def load_event_data() -> pd.DataFrame:
    query = "SELECT * FROM events ORDER BY start_ts"
    return pd.read_sql(query, ENGINE, parse_dates=["start_ts", "end_ts"])


def compute_quality_metrics(df: pd.DataFrame) -> dict:
    tags = df["tag"].unique()
    report = {
        "total_rows": int(len(df)),
        "total_tags": int(len(tags)),
        "tags": {},
    }

    for tag in sorted(tags):
        tag_df = df[df["tag"] == tag].copy()
        total = len(tag_df)
        missing_pct = float(tag_df["value"].isna().sum() / total * 100)
        clean = tag_df.dropna(subset=["value"])

        if len(clean) > 0:
            z_scores = np.abs(stats.zscore(clean["value"]))
            outliers_z = int((z_scores > 3).sum())

            q1, q3 = np.percentile(clean["value"], [25, 75])
            iqr = q3 - q1
            outliers_iqr = int(
                (
                    (clean["value"] < q1 - 1.5 * iqr)
                    | (clean["value"] > q3 + 1.5 * iqr)
                ).sum()
            )

            if len(clean) >= 50:
                adf_pvalue = float(adfuller(clean["value"].values)[1])
            else:
                adf_pvalue = None

            gaps = clean["timestamp"].diff()
            large_gaps = int((gaps > pd.Timedelta("2min")).sum())
        else:
            outliers_z = 0
            outliers_iqr = 0
            adf_pvalue = None
            large_gaps = 0

        report["tags"][tag] = {
            "count": int(total),
            "missing_percentage": round(missing_pct, 3),
            "outliers_z_score": outliers_z,
            "outliers_iqr": outliers_iqr,
            "adf_p_value": None if adf_pvalue is None else round(adf_pvalue, 4),
            "gaps_over_2min": large_gaps,
        }

    return report


def compute_event_summary(events: pd.DataFrame) -> dict:
    summary = {
        "total_events": int(len(events)),
        "by_event_type": events["event_type"].value_counts().to_dict(),
        "by_error_kind": events["error_kind"].value_counts().to_dict(),
    }
    return summary


def save_report(report: dict, filename: str = "dq_report_db.json") -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print("Carregando dados do banco...")
    sensor_df = load_sensor_data()
    event_df = load_event_data()

    print(
        f"Leitura carregada: {len(sensor_df)} linhas, {sensor_df['tag'].nunique()} tags"
    )
    report = compute_quality_metrics(sensor_df)
    report["events"] = compute_event_summary(event_df)

    save_report(report)
    print("✓ Relatório de qualidade de dados salvo em 'dq_report_db.json'")
