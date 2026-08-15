#!/bin/bash
# Tạo thêm database "nps_app" ngoài database "airflow" mặc định,
# để dùng chung 1 Postgres instance cho cả Airflow metadata và app data.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE nps_app;
EOSQL
