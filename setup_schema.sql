CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Dropar tabelas existentes
DROP TABLE IF EXISTS events CASCADE;

DROP TABLE IF EXISTS sensor_readings CASCADE;

-- Tabela para leituras de sensores no formato narrow
CREATE TABLE sensor_readings (
    timestamp TIMESTAMPTZ NOT NULL,
    tag TEXT NOT NULL,
    value DOUBLE PRECISION,
    unit TEXT,
    mode TEXT,
    burner_cmd DOUBLE PRECISION,
    load_factor DOUBLE PRECISION,
    door_open INTEGER,
    fan_cmd DOUBLE PRECISION,
    error_type TEXT,
    quality TEXT
);

SELECT create_hypertable (
        'sensor_readings', 'timestamp', chunk_time_interval = > INTERVAL '1 day'
    );

CREATE INDEX ON sensor_readings (tag, timestamp DESC);

-- Tabela para eventos
CREATE TABLE events (
    event_type TEXT,
    signal TEXT,
    error_kind TEXT,
    start_ts TIMESTAMPTZ,
    end_ts TIMESTAMPTZ,
    magnitude DOUBLE PRECISION
);

SELECT create_hypertable (
        'events', 'start_ts', chunk_time_interval = > INTERVAL '1 day'
    );