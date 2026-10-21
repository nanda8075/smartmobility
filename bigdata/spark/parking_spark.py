
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

ROOT = Path(__file__).resolve().parents[2]
PARKING_PATH = ROOT / "bigdata" / "datasets" / "parking_data.csv"
OUTPUT_PATH = ROOT / "bigdata" / "output" / "parking_analytics"
OUTPUT_FILE = OUTPUT_PATH / "parking_analytics.csv"

spark = (
    SparkSession.builder
    .appName("SmartMobilityParkingAnalytics")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # Read parking dataset
    parking = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(PARKING_PATH))
    )

    # Calculate available spaces and occupancy percentage
    parking = (
        parking
        .withColumn("capacity", col("capacity").cast("double"))
        .withColumn("occupied", col("occupied").cast("double"))
        .withColumn(
            "available",
            when(
                col("capacity") - col("occupied") < 0, 0
            ).otherwise(col("capacity") - col("occupied"))
        )
        .withColumn(
            "occupancy_pct",
            when(
                col("capacity") > 0,
                col("occupied") / col("capacity") * 100
            ).otherwise(0)
        )
    )

    print("Parking occupancy by location:")

    summary = parking.groupBy("location").agg(
        {"capacity": "sum", "occupied": "sum", "available": "sum"}
    )
    summary.show(truncate=False)

    # Save the result as a single CSV file
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    parking.toPandas().to_csv(OUTPUT_FILE, index=False)

    print("Parking analytics completed successfully!")
    print(f"Output saved to: {OUTPUT_FILE}")

finally:
    spark.stop()
