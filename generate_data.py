from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Iterable
import numpy as np
import pandas as pd


ErrorType = Literal["none", "spike", "drift", "stuck", "missing"]
ModeType = Literal["warmup", "steady", "load_change", "disturbance", "shutdown"]


@dataclass
class ZoneConfig:
    name: str
    base_setpoint: float
    setpoint_variation: float = 35.0
    tau_zone_min: float = 18.0
    tau_load_min: float = 55.0
    zone_sensor_std: float = 1.8
    load_sensor_std: float = 1.2
    load_offset_base: float = 18.0


@dataclass
class DraftConfig:
    nominal: float = -18.0
    tau_min: float = 2.5
    sensor_std: float = 0.35


@dataclass
class FurnaceV2Config:
    start: str = "2026-01-01 00:00:00"
    end: str = "2026-01-03 00:00:00"
    freq: str = "1min"
    ambient_temp: float = 28.0
    error_rate: float = 0.02
    seed: int = 42

    zones: list[ZoneConfig] = field(
        default_factory=lambda: [
            ZoneConfig(
                name="01",
                base_setpoint=820.0,
                setpoint_variation=30.0,
                tau_zone_min=14.0,
                tau_load_min=45.0,
            ),
            ZoneConfig(
                name="02",
                base_setpoint=900.0,
                setpoint_variation=35.0,
                tau_zone_min=18.0,
                tau_load_min=60.0,
            ),
            ZoneConfig(
                name="03",
                base_setpoint=980.0,
                setpoint_variation=40.0,
                tau_zone_min=22.0,
                tau_load_min=75.0,
            ),
        ]
    )

    draft: DraftConfig = field(default_factory=DraftConfig)

    include_true_values: bool = True
    include_quality_flags: bool = True
    include_event_table: bool = True


class FurnaceDataGeneratorV2:
    def __init__(self, config: FurnaceV2Config) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

    # ========= API principal =========

    def generate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retorna:
        - df_wide: tabela principal, uma linha por timestamp
        - df_events: tabela de eventos/erros
        """
        idx = pd.date_range(
            self.cfg.start, self.cfg.end, freq=self.cfg.freq, inclusive="left"
        )
        if len(idx) < 10:
            raise ValueError("A faixa temporal gerou poucos pontos.")

        n = len(idx)
        dt_min = pd.to_timedelta(self.cfg.freq).total_seconds() / 60.0

        common = self._build_common_profile(n)
        df = pd.DataFrame({"timestamp": idx})
        df["mode"] = common["mode"]
        df["burner_cmd"] = common["burner_cmd"]
        df["load_factor"] = common["load_factor"]
        df["door_open"] = common["door_open"].astype(int)
        df["fan_cmd"] = common["fan_cmd"]

        # gera cada zona
        zone_true_map: dict[str, np.ndarray] = {}
        load_true_map: dict[str, np.ndarray] = {}

        for i, zcfg in enumerate(self.cfg.zones):
            setpoint = self._build_zone_setpoint(n, zcfg, common["mode"])
            t_zone_true, t_load_true = self._simulate_zone(
                zcfg=zcfg,
                n=n,
                dt_min=dt_min,
                setpoint=setpoint,
                burner_cmd=common["burner_cmd"],
                load_factor=common["load_factor"],
                door_open=common["door_open"],
                upstream_zone=(
                    None if i == 0 else zone_true_map[self.cfg.zones[i - 1].name]
                ),
            )

            zone_true_map[zcfg.name] = t_zone_true
            load_true_map[zcfg.name] = t_load_true

            # observados
            t_zone_obs = t_zone_true + self.rng.normal(
                0.0, zcfg.zone_sensor_std, size=n
            )
            t_load_obs = t_load_true + self.rng.normal(
                0.0, zcfg.load_sensor_std, size=n
            )

            df[f"SP_Zone_{zcfg.name}"] = setpoint
            df[f"T_Zone_{zcfg.name}"] = t_zone_obs
            df[f"T_Load_{zcfg.name}"] = t_load_obs

            if self.cfg.include_true_values:
                df[f"T_Zone_{zcfg.name}_true"] = t_zone_true
                df[f"T_Load_{zcfg.name}_true"] = t_load_true

        # pressão de draft usa visão global do forno
        p_draft_true = self._simulate_draft(
            n=n,
            dt_min=dt_min,
            fan_cmd=common["fan_cmd"],
            burner_cmd=common["burner_cmd"],
            load_factor=common["load_factor"],
            door_open=common["door_open"],
            hot_zone_temp=zone_true_map[self.cfg.zones[-1].name],
        )
        p_draft_obs = p_draft_true + self.rng.normal(
            0.0, self.cfg.draft.sensor_std, size=n
        )

        df["P_Draft"] = p_draft_obs
        if self.cfg.include_true_values:
            df["P_Draft_true"] = p_draft_true

        # tabela de eventos
        events = []

        # injeta erros em todas as séries principais
        signal_columns = []
        for zcfg in self.cfg.zones:
            signal_columns.extend(
                [
                    f"T_Zone_{zcfg.name}",
                    f"T_Load_{zcfg.name}",
                ]
            )
        signal_columns.append("P_Draft")

        df, error_events = self._inject_errors(df, signal_columns)
        events.extend(error_events)

        # flags de qualidade
        if self.cfg.include_quality_flags:
            for col in signal_columns:
                err_col = f"{col}_error"
                df[f"{col}_quality"] = np.where(df[err_col] == "none", "good", "bad")

        # eventos operacionais
        if self.cfg.include_event_table:
            events.extend(self._extract_operational_events(df))

        df_events = pd.DataFrame(events)
        if not df_events.empty:
            df_events = df_events.sort_values(
                by=["start_ts", "signal"], na_position="last"
            ).reset_index(drop=True)

        return df, df_events

    def to_narrow(self, df_wide: pd.DataFrame) -> pd.DataFrame:
        """
        Converte a saída wide para narrow.
        Não grava em banco, só prepara a estrutura.
        """
        value_cols = [c for c in df_wide.columns if self._is_signal_column(c)]
        rows = []

        meta_cols = [
            "timestamp",
            "mode",
            "burner_cmd",
            "load_factor",
            "door_open",
            "fan_cmd",
        ]

        for col in value_cols:
            error_col = f"{col}_error" if f"{col}_error" in df_wide.columns else None
            quality_col = (
                f"{col}_quality" if f"{col}_quality" in df_wide.columns else None
            )

            base = df_wide[meta_cols + [col]].copy()
            base["tag"] = col
            base["value"] = base[col]
            base.drop(columns=[col], inplace=True)

            if error_col:
                base["error_type"] = df_wide[error_col].values
            else:
                base["error_type"] = "none"

            if quality_col:
                base["quality"] = df_wide[quality_col].values
            else:
                base["quality"] = "good"

            base["unit"] = self._infer_unit(col)
            rows.append(base)

        narrow = pd.concat(rows, ignore_index=True)
        return narrow

    # ========= perfis operacionais =========

    def _build_common_profile(self, n: int) -> dict[str, np.ndarray]:
        mode = np.full(n, "steady", dtype=object)

        warmup_len = max(30, int(0.12 * n))
        mode[:warmup_len] = "warmup"

        # shutdown opcional perto do final
        if n > 300 and self.rng.random() < 0.35:
            shut_start = int(self.rng.integers(int(n * 0.82), int(n * 0.95)))
            mode[shut_start:] = "shutdown"

        # eventos intermediários
        event_count = max(3, n // 700)
        for _ in range(event_count):
            s = self.rng.integers(warmup_len, max(warmup_len + 1, n - 60))
            dur = int(self.rng.integers(20, 120))
            e = min(n, s + dur)
            if np.all(mode[s:e] != "shutdown"):
                mode[s:e] = self.rng.choice(["load_change", "disturbance"])

        # burner_cmd
        burner_cmd = np.zeros(n, dtype=float)
        burner_cmd[mode == "warmup"] = np.linspace(0.45, 0.92, np.sum(mode == "warmup"))
        burner_cmd[mode == "steady"] = 0.72 + self.rng.normal(
            0.0, 0.03, size=np.sum(mode == "steady")
        )
        burner_cmd[mode == "load_change"] = 0.78 + self.rng.normal(
            0.0, 0.05, size=np.sum(mode == "load_change")
        )
        burner_cmd[mode == "disturbance"] = 0.66 + self.rng.normal(
            0.0, 0.08, size=np.sum(mode == "disturbance")
        )
        burner_cmd[mode == "shutdown"] = (
            np.linspace(0.35, 0.02, np.sum(mode == "shutdown"))
            if np.sum(mode == "shutdown") > 0
            else burner_cmd[mode == "shutdown"]
        )
        burner_cmd = np.clip(burner_cmd, 0.0, 1.0)

        # fan_cmd
        fan_cmd = np.clip(
            0.68 + 0.18 * burner_cmd + self.rng.normal(0.0, 0.03, size=n), 0.25, 1.0
        )

        # load_factor
        load_factor = np.clip(
            0.85
            + 0.18 * np.sin(np.linspace(0, 8 * np.pi, n))
            + self.rng.normal(0.0, 0.05, size=n),
            0.45,
            1.45,
        )
        load_factor[mode == "load_change"] += self.rng.normal(
            0.12, 0.05, size=np.sum(mode == "load_change")
        )
        load_factor[mode == "shutdown"] *= 0.35
        load_factor = np.clip(load_factor, 0.2, 1.6)

        # door_open
        door_open = np.zeros(n, dtype=bool)
        door_events = max(4, n // 1000)
        for _ in range(door_events):
            s = self.rng.integers(warmup_len, max(warmup_len + 1, n - 10))
            dur = int(self.rng.integers(2, 10))
            door_open[s : min(n, s + dur)] = True

        return {
            "mode": mode,
            "burner_cmd": burner_cmd,
            "fan_cmd": fan_cmd,
            "load_factor": load_factor,
            "door_open": door_open,
        }

    def _build_zone_setpoint(
        self, n: int, zcfg: ZoneConfig, mode: np.ndarray
    ) -> np.ndarray:
        sp = np.full(n, zcfg.base_setpoint, dtype=float)

        warmup_mask = mode == "warmup"
        warmup_n = int(np.sum(warmup_mask))
        if warmup_n > 0:
            sp[warmup_mask] = np.linspace(
                self.cfg.ambient_temp + 30.0, zcfg.base_setpoint, warmup_n
            )

        current = zcfg.base_setpoint
        pos = warmup_n
        while pos < n:
            block = int(self.rng.integers(100, 420))
            current += self.rng.normal(0.0, zcfg.setpoint_variation / 3.0)
            current = float(
                np.clip(
                    current,
                    zcfg.base_setpoint - zcfg.setpoint_variation,
                    zcfg.base_setpoint + zcfg.setpoint_variation,
                )
            )
            sp[pos : min(n, pos + block)] = current
            pos += block

        sp[mode == "disturbance"] += self.rng.normal(
            -10.0, 7.0, size=np.sum(mode == "disturbance")
        )
        sp[mode == "shutdown"] = (
            np.linspace(
                (
                    float(sp[mode == "shutdown"][0])
                    if np.sum(mode == "shutdown") > 0
                    else zcfg.base_setpoint
                ),
                self.cfg.ambient_temp + 40.0,
                np.sum(mode == "shutdown"),
            )
            if np.sum(mode == "shutdown") > 0
            else sp[mode == "shutdown"]
        )

        return sp

    # ========= processo =========

    def _simulate_zone(
        self,
        zcfg: ZoneConfig,
        n: int,
        dt_min: float,
        setpoint: np.ndarray,
        burner_cmd: np.ndarray,
        load_factor: np.ndarray,
        door_open: np.ndarray,
        upstream_zone: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        t_zone = np.zeros(n, dtype=float)
        t_load = np.zeros(n, dtype=float)

        t_zone[0] = self.cfg.ambient_temp + 25.0
        t_load[0] = self.cfg.ambient_temp + 25.0

        alpha_zone = dt_min / max(dt_min, zcfg.tau_zone_min)
        alpha_load = dt_min / max(dt_min, zcfg.tau_load_min)

        zone_noise = self._ar1(n, phi=0.92, sigma=0.85)
        load_noise = self._ar1(n, phi=0.96, sigma=0.30)

        for i in range(1, n):
            coupling = 0.0
            if upstream_zone is not None:
                coupling = 0.06 * (upstream_zone[i - 1] - t_zone[i - 1])

            target_zone = (
                setpoint[i]
                + 22.0 * (burner_cmd[i] - 0.5)
                - 20.0 * (load_factor[i] - 1.0)
                - 28.0 * float(door_open[i])
                + coupling
            )

            t_zone[i] = (
                t_zone[i - 1]
                + alpha_zone * (target_zone - t_zone[i - 1])
                + zone_noise[i]
            )

            target_load = t_zone[i] - (
                zcfg.load_offset_base + 9.0 * max(0.0, load_factor[i] - 0.9)
            )
            t_load[i] = (
                t_load[i - 1]
                + alpha_load * (target_load - t_load[i - 1])
                + load_noise[i]
            )

        return t_zone, t_load

    def _simulate_draft(
        self,
        n: int,
        dt_min: float,
        fan_cmd: np.ndarray,
        burner_cmd: np.ndarray,
        load_factor: np.ndarray,
        door_open: np.ndarray,
        hot_zone_temp: np.ndarray,
    ) -> np.ndarray:
        p = np.zeros(n, dtype=float)
        p[0] = self.cfg.draft.nominal

        alpha = dt_min / max(dt_min, self.cfg.draft.tau_min)
        noise = self._ar1(n, phi=0.80, sigma=0.22)

        for i in range(1, n):
            temp_effect = 0.0035 * (hot_zone_temp[i] - np.nanmean(hot_zone_temp))
            target = (
                self.cfg.draft.nominal
                - 4.8 * (fan_cmd[i] - 0.5)
                + 2.6 * float(door_open[i])
                + 1.4 * (load_factor[i] - 1.0)
                - 1.0 * (burner_cmd[i] - 0.5)
                + temp_effect
            )

            p[i] = p[i - 1] + alpha * (target - p[i - 1]) + noise[i]

        return p

    def _ar1(self, n: int, phi: float, sigma: float) -> np.ndarray:
        x = np.zeros(n, dtype=float)
        eps = self.rng.normal(0.0, sigma, size=n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + eps[i]
        return x

    # ========= erros =========

    def _inject_errors(
        self,
        df: pd.DataFrame,
        signal_columns: Iterable[str],
    ) -> tuple[pd.DataFrame, list[dict]]:
        n = len(df)
        events: list[dict] = []

        for signal in signal_columns:
            df[f"{signal}_error"] = "none"

        total_bad_points = int(n * self.cfg.error_rate)
        if total_bad_points <= 0:
            return df, events

        signal_columns = list(signal_columns)
        error_types: list[ErrorType] = ["spike", "drift", "stuck", "missing"]

        for _ in range(total_bad_points):
            signal = self.rng.choice(signal_columns)
            kind = self.rng.choice(error_types, p=[0.38, 0.20, 0.17, 0.25])
            idx = int(self.rng.integers(0, n))

            if kind == "spike":
                mag = self._spike_magnitude(signal)
                df.at[idx, signal] = df.at[idx, signal] + mag
                df.at[idx, f"{signal}_error"] = "spike"
                events.append(
                    {
                        "event_type": "data_error",
                        "signal": signal,
                        "error_kind": "spike",
                        "start_ts": df.at[idx, "timestamp"],
                        "end_ts": df.at[idx, "timestamp"],
                        "magnitude": mag,
                    }
                )

            elif kind == "missing":
                span = int(self.rng.integers(1, 10))
                end = min(n, idx + span)
                df.loc[idx : end - 1, signal] = np.nan
                df.loc[idx : end - 1, f"{signal}_error"] = "missing"
                events.append(
                    {
                        "event_type": "data_error",
                        "signal": signal,
                        "error_kind": "missing",
                        "start_ts": df.at[idx, "timestamp"],
                        "end_ts": df.at[end - 1, "timestamp"],
                        "magnitude": None,
                    }
                )

            elif kind == "stuck":
                span = int(self.rng.integers(5, 50))
                end = min(n, idx + span)
                stuck_val = float(df.at[idx, signal])
                df.loc[idx : end - 1, signal] = stuck_val
                df.loc[idx : end - 1, f"{signal}_error"] = "stuck"
                events.append(
                    {
                        "event_type": "data_error",
                        "signal": signal,
                        "error_kind": "stuck",
                        "start_ts": df.at[idx, "timestamp"],
                        "end_ts": df.at[end - 1, "timestamp"],
                        "magnitude": stuck_val,
                    }
                )

            elif kind == "drift":
                span = int(self.rng.integers(20, 140))
                end = min(n, idx + span)
                mag = self._drift_total(signal)
                drift = np.linspace(0.0, mag, end - idx)
                df.loc[idx : end - 1, signal] = (
                    df.loc[idx : end - 1, signal].to_numpy() + drift
                )
                df.loc[idx : end - 1, f"{signal}_error"] = "drift"
                events.append(
                    {
                        "event_type": "data_error",
                        "signal": signal,
                        "error_kind": "drift",
                        "start_ts": df.at[idx, "timestamp"],
                        "end_ts": df.at[end - 1, "timestamp"],
                        "magnitude": mag,
                    }
                )

        return df, events

    def _spike_magnitude(self, signal: str) -> float:
        if signal.startswith("T_Zone"):
            return float(self.rng.normal(0.0, 35.0))
        if signal.startswith("T_Load"):
            return float(self.rng.normal(0.0, 20.0))
        return float(self.rng.normal(0.0, 4.0))

    def _drift_total(self, signal: str) -> float:
        sign = self.rng.choice([-1, 1])
        if signal.startswith("T_Zone"):
            return float(sign * self.rng.normal(16.0, 7.0))
        if signal.startswith("T_Load"):
            return float(sign * self.rng.normal(10.0, 4.0))
        return float(sign * self.rng.normal(1.6, 0.6))

    # ========= utilitários =========

    def _extract_operational_events(self, df: pd.DataFrame) -> list[dict]:
        events = []

        mode = df["mode"].to_numpy()
        ts = df["timestamp"].to_numpy()

        start = 0
        for i in range(1, len(mode) + 1):
            if i == len(mode) or mode[i] != mode[start]:
                if mode[start] != "steady":
                    events.append(
                        {
                            "event_type": "operation_mode",
                            "signal": None,
                            "error_kind": mode[start],
                            "start_ts": ts[start],
                            "end_ts": ts[i - 1],
                            "magnitude": None,
                        }
                    )
                start = i

        return events

    def _is_signal_column(self, col: str) -> bool:
        if col.startswith("T_Zone_"):
            return (
                not col.endswith("_true")
                and not col.endswith("_error")
                and not col.endswith("_quality")
            )
        if col.startswith("T_Load_"):
            return (
                not col.endswith("_true")
                and not col.endswith("_error")
                and not col.endswith("_quality")
            )
        if col == "P_Draft":
            return True
        return False

    def _infer_unit(self, col: str) -> str:
        if col.startswith("T_"):
            return "degC"
        if col.startswith("P_"):
            return "Pa"
        return "unknown"


# ========= interface simples =========


def generate_furnace_v2(
    start: str,
    end: str,
    freq: str = "1min",
    error_rate: float = 0.02,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = FurnaceV2Config(
        start=start,
        end=end,
        freq=freq,
        error_rate=error_rate,
        seed=seed,
    )

    generator = FurnaceDataGeneratorV2(cfg)
    df_wide, df_events = generator.generate()
    df_narrow = generator.to_narrow(df_wide)

    return df_wide, df_narrow, df_events


def _apply_physical_limits(self, df):

    temp_cols = [c for c in df.columns if c.startswith("T_")]

    for col in temp_cols:
        df[col] = df[col].clip(lower=self.cfg.ambient_temp - 5, upper=1400)

    if "P_Draft" in df.columns:
        df["P_Draft"] = df["P_Draft"].clip(lower=-60, upper=10)

    return df


if __name__ == "__main__":
    df_wide, df_narrow, df_events = generate_furnace_v2(
        start="2026-01-01 00:00:00",
        end="2026-01-05 00:00:00",
        freq="1min",
        error_rate=0.03,
        seed=7,
    )

    print("WIDE")
    print(df_wide.head(), "\n")

    print("NARROW")
    print(df_narrow.head(), "\n")

    print("EVENTS")
    print(df_events.head(), "\n")

    df_wide.to_csv("furnace_v2_wide.csv", index=False)
    df_narrow.to_csv("furnace_v2_narrow.csv", index=False)
    df_events.to_csv("furnace_v2_events.csv", index=False)
