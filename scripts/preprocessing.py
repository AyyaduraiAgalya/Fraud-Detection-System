from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql.functions import col
from pyspark.sql import types as T
from data_loading import load_data

def ensure_numeric_columns(df: DataFrame, numerical_cols: list) -> DataFrame:
    """
    Ensures that columns intended for numerical processing are of a numeric type.
    Converts columns to DoubleType if they are not already numeric.
    """
    for col_name in numerical_cols:
        if not isinstance(df.schema[col_name].dataType, T.NumericType):
            df = df.withColumn(col_name, col(col_name).cast(T.DoubleType()))
    return df


def scale_features(df: DataFrame) -> DataFrame:
    """
    Scales numerical features using StandardScaler.
    """
    # Define numerical columns excluding the target variable
    numerical_cols = [col for col in df.columns if col != "Class"]  # Assuming "Class" is the target

    # Ensure all numerical columns are of numeric data type
    df = ensure_numeric_columns(df, numerical_cols)

    # Assemble all numerical columns into a single feature vector
    assembler = VectorAssembler(inputCols=numerical_cols, outputCol="features")
    df = assembler.transform(df)

    # Apply Standard Scaling
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
    df = scaler.fit(df).transform(df)

    # Select only the scaled features and the target column
    df = df.select("scaled_features", "Class")
    return df


def split_data(df: DataFrame, train_ratio: float = 0.8):
    """
    Splits the data into training and test sets.
    """
    return df.randomSplit([train_ratio, 1 - train_ratio], seed=42)


def main_preprocessing(input_path: str, output_path: str):
    """
    Main preprocessing function to load, scale, and split data.
    """
    # Load engineered data
    df = load_data(input_path)

    # Scale numerical features
    df = scale_features(df)

    # Split the data into train and test sets
    train_df, test_df = split_data(df)

    # Save preprocessed data
    train_df.write.mode("overwrite").parquet(output_path + "/train")
    test_df.write.mode("overwrite").parquet(output_path + "/test")

    print("Preprocessing complete. Data saved to:", output_path)


# Example Usage
if __name__ == "__main__":
    engineered_data_path = "data/engineered/engineered_data"
    preprocessed_data_path = "data/preprocessed/preprocessed_data"
    main_preprocessing(engineered_data_path, preprocessed_data_path)
