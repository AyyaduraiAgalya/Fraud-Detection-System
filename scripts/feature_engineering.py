from pyspark.sql import Window
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, avg, stddev, lag


def add_transaction_count_features(df: DataFrame) -> DataFrame:
    """
    Adds transaction frequency features: count in the last 1 hour and last 5 mins.
    """
    one_hour_window = Window.orderBy("Time").rangeBetween(-3600, 0)  # Last 1 hour
    five_mins_window = Window.orderBy("Time").rangeBetween(-300, 0)  # Last 5 mins

    df = df.withColumn("Transactions_Last_1_Hour", count("Amount").over(one_hour_window))
    df = df.withColumn("Transactions_Last_5_Mins", count("Amount").over(five_mins_window))
    return df


def add_avg_amount_feature(df: DataFrame) -> DataFrame:
    """
    Adds a feature for the average transaction amount in the last 1 hour and 5 mins..
    """
    one_hour_window = Window.orderBy("Time").rangeBetween(-3600, 0)
    five_mins_window = Window.orderBy("Time").rangeBetween(-300, 0)

    df = df.withColumn("Avg_Amount_Last_1_Hour", avg("Amount").over(one_hour_window))
    df = df.withColumn("Avg_Amount_Last_5_Mins", avg("Amount").over(five_mins_window))
    return df

def add_high_amount_flag(df: DataFrame, threshold: float = 2000) -> DataFrame:
    """
    Adds a binary flag for high-value transactions based on a specified threshold. Defaulting threshold to 2000
    """
    df = df.withColumn("High_Amount", (col("Amount") > threshold).cast("integer"))
    return df


def add_stddev_amount_feature(df: DataFrame) -> DataFrame:
    """
    Adds a feature for the rolling standard deviation of transaction amounts in the last 1 hour and 5 mins..
    """
    one_hour_window = Window.orderBy("Time").rangeBetween(-3600, 0)
    five_mins_window = Window.orderBy("Time").rangeBetween(-300, 0)
    df = df.withColumn("Stddev_Amount_Last_1_Hour", stddev("Amount").over(one_hour_window))
    df = df.withColumn("Stddev_Amount_Last_5_Mins", stddev("Amount").over(five_mins_window))
    return df


def apply_feature_engineering(df: DataFrame) -> DataFrame:
    """
    Applies all feature engineering functions to the input DataFrame.
    """
    df = add_transaction_count_features(df)
    df = add_avg_amount_feature(df)
    df = add_high_amount_flag(df)
    df = add_stddev_amount_feature(df)
    return df
