CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE sensor_readings (
    time TIMESTAMPTZ NOT NULL,
    sensor_id TEXT NOT NULL,
    pressure DOUBLE PRECISION, -- bar
    temperature DOUBLE PRECISION -- celsius
);

SELECT create_hypertable (
        'sensor_readings', 'time', chunk_time_interval = > INTERVAL '1 day'
    );

CREATE INDEX ON sensor_readings (sensor_id, time DESC);

SELECT
    time_bucket ('1 hour', time) AS bucket,
    sensor_id,
    AVG(pressure) AS avg_pressure,
    MAX(temperature) AS max_temperature STDDEV(temperature) AS stddev_temperature
FROM sensor_readings
WHERE
    time >= NOW() - INTERVAL '7 days'
GROUP BY
    bucket,
    sensor_id
ORDER BY bucket;