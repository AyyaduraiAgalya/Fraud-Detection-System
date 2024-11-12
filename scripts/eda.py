from pyspark.sql import SparkSession
from pyspark.sql.functions import count, when, col

def perform_eda(df):
    # Summary statistics for all numerical columns
    print("Summary Statistics:")
    df.describe().show()

    # Count of missing values per column
    print("Missing Values:")
    missing_values = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
    missing_values.show()

    # Class distribution (fraud vs non-fraud)
    print("Class Distribution (Fraud vs Non-Fraud):")
    df.groupBy("Class").count().show()

if __name__ == "__main__":
    # Initialising Spark session and load data
    spark = SparkSession.builder.appName("EDA").getOrCreate()
    data_path = "data/raw/creditcard.csv"
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    # Performing EDA
    perform_eda(df)
