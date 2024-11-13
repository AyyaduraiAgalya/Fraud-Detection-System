from pyspark.sql import Window, DataFrame
from pyspark.sql.functions import col, count, avg, stddev
from data_loading import load_data

def add_transaction_count_features(df: DataFrame) -> DataFrame:
    """Adds transaction frequency features: count in the last 1 hour and last 5 mins."""
    one_hour_window = Window.orderBy("Time").rangeBetween(-3600, 0)
    five_mins_window = Window.orderBy("Time").rangeBetween(-300, 0)

    df = df.withColumn("Transactions_Last_1_Hour", count("Amount").over(one_hour_window))
    df = df.withColumn("Transactions_Last_5_Mins", count("Amount").over(five_mins_window))
    return df

def add_avg_amount_feature(df: DataFrame) -> DataFrame:
    """Adds a feature for the average transaction amount in the last 1 hour and 5 mins."""
    one_hour_window = Window.orderBy("Time").rangeBetween(-3600, 0)
    five_mins_window = Window.orderBy("Time").rangeBetween(-300, 0)

    df = df.withColumn("Avg_Amount_Last_1_Hour", avg("Amount").over(one_hour_window))
    df = df.withColumn("Avg_Amount_Last_5_Mins", avg("Amount").over(five_mins_window))
    return df

def add_high_amount_flag(df: DataFrame, threshold: float = 2000) -> DataFrame:
    """Adds a binary flag for high-value transactions based on a specified threshold."""
    df = df.withColumn("High_Amount", (col("Amount") > threshold).cast("integer"))
    return df

def add_stddev_amount_feature(df: DataFrame) -> DataFrame:
    """Adds a feature for the rolling standard deviation of transaction amounts in the last 1 hour and 5 mins."""
    one_hour_window = Window.orderBy("Time").rangeBetween(-3600, 0)
    five_mins_window = Window.orderBy("Time").rangeBetween(-300, 0)
    df = df.withColumn("Stddev_Amount_Last_1_Hour", stddev("Amount").over(one_hour_window))
    df = df.withColumn("Stddev_Amount_Last_5_Mins", stddev("Amount").over(five_mins_window))
    return df

def main_feature_engineering(input_path: str, output_path: str) -> None:
    """Main function to perform feature engineering on the dataset."""
    # Loading raw data
    df = load_data(input_path)

    # Apply feature engineering functions
    df = add_transaction_count_features(df)
    df = add_avg_amount_feature(df)
    df = add_high_amount_flag(df)
    df = add_stddev_amount_feature(df)

    # Save engineered data for preprocessing
    df.coalesce(1).write.csv(output_path, header=True, mode="overwrite")

# Example Usage
if __name__ == "__main__":
    raw_data_path = "data/raw/creditcard.csv"
    engineered_data_path = "data/engineered"
    main_feature_engineering(raw_data_path, engineered_data_path)