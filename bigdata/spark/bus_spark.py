from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, unix_timestamp, when

ROOT = Path(__file__).resolve().parents[2]
BUS_PATH = ROOT / "bigdata" / "datasets" / "bus_data.csv"
OUTPUT_PATH = ROOT / "bigdata" / "output" / "bus_analytics"
import os
os.environ["HADOOP_HOME"] = r"C:\hadoop"
spark = (
    SparkSession.builder
    .appName("SmartMobilityBusAnalytics")
    .master("local[1]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

bus = spark.read.option("header", True).option(
    "inferSchema", True
).csv(str(BUS_PATH))

bus = bus.withColumn(
    "scheduled_ts",
    to_timestamp(col("scheduled_time"), "HH:mm")
).withColumn(
    "actual_ts",
    to_timestamp(col("actual_time"), "HH:mm")
)

bus = bus.withColumn(
    "delay_minutes",
    when(
        col("scheduled_ts").isNotNull()
        & col("actual_ts").isNotNull(),
        (unix_timestamp("actual_ts") -
         unix_timestamp("scheduled_ts")) / 60
    ).otherwise(0)
)

bus = bus.withColumn(
    "delay_minutes",
    when(col("delay_minutes") < 0, 0)
    .otherwise(col("delay_minutes"))
)

bus.groupBy("route").avg("delay_minutes").show(
    truncate=False
)

output_dir = OUTPUT_PATH.parent
output_dir.mkdir(parents=True, exist_ok=True)

import csv

result = bus.toPandas()
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

result.to_csv(
    str(OUTPUT_PATH / "bus_analytics.csv"),
    index=False
)

print("Bus analytics completed successfully!")
print(f"Output saved to: {OUTPUT_PATH}")

print("Bus analytics completed successfully!")
print(f"Output saved to: {OUTPUT_PATH}")

spark.stop()
