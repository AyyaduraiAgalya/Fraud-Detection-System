from pyspark.sql import SparkSession
from feature_engineering import apply_feature_engineering

# Initialising Spark session
spark = SparkSession.builder.appName("FraudDetectionFeatureEngineering").getOrCreate()

# Loading the data
data_path = "data/raw/creditcard.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Applying feature engineering
df_with_features = apply_feature_engineering(df)

# Saving the transformed DataFrame
output_path = "data/processed/processed_creditcard.csv"
df_with_features.write.csv(output_path, header=True, mode="overwrite")

# Looking into the sample of engineered data
df_with_features.show(10)
