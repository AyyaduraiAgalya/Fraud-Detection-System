from pyspark.sql import SparkSession

def load_data(file_path):
    # Initialising Spark session
    spark = SparkSession.builder.appName("Credit Card Fraud Detection").getOrCreate()

    # Loading the dataset
    df = spark.read.csv(file_path, header=True, inferSchema=True)

    # Displaying schema to verify correct data types
    df.printSchema()
    return df

if __name__ == "__main__":
    # Path to the dataset
    data_path = "data/raw/creditcard.csv"
    df = load_data(data_path)
    # First few rows for verification
    df.show(5)
