"""Spark session tuned for a single-machine ASA run."""
from __future__ import annotations

import os

from pyspark.sql import SparkSession


def build_session(app_name: str = "asa",
                  driver_memory: str = "8g",
                  shuffle_partitions: int = 96,
                  local_dir: str | None = None) -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "200000")
        .config("spark.driver.maxResultSize", "4g")
    )
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)
        builder = builder.config("spark.local.dir", local_dir)
    return builder.getOrCreate()
