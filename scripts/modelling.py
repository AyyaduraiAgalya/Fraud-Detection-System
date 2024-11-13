from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
import os


def load_preprocessed_data(train_path: str, test_path: str):
    """
    Loads preprocessed train and test data.
    """
    spark = SparkSession.builder.appName("FraudDetectionModel").getOrCreate()
    train_df = spark.read.parquet(train_path)
    test_df = spark.read.parquet(test_path)
    return train_df, test_df


def train_logistic_regression(train_df):
    """
    Trains a logistic regression model.
    """
    # Initialising the logistic regression model
    lr = LogisticRegression(featuresCol="scaled_features", labelCol="Class", predictionCol="prediction")

    # Creating pipeline with the logistic regression stage
    pipeline = Pipeline(stages=[lr])

    # Fitting the pipeline to the training data
    model = pipeline.fit(train_df)
    return model


def evaluate_model(model, test_df):
    """
    Evaluates the model on the test set.
    """
    # Making predictions on the test set
    predictions = model.transform(test_df)

    # Using AUC as the evaluation metric
    evaluator = BinaryClassificationEvaluator(labelCol="Class", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    print(f"Test AUC: {auc:.4f}")

    return auc


def main_modeling(train_path: str, test_path: str, model_output_path: str, use_cache=True):
    """
    Main modeling function to train the model and save output.
    Skips processing if model already exists.
    """
    if use_cache:
        # Checking if model already exists
        if os.path.exists(model_output_path):
            print("Model already exists. Skipping modelling step.")
            return
    else:
        # Loading preprocessed data
        train_df, test_df = load_preprocessed_data(train_path, test_path)

        # Training the model
        model = train_logistic_regression(train_df)

        # Evaluating the model
        auc = evaluate_model(model, test_df)

        # Saving the model
        model.write().overwrite().save(model_output_path)
        print("Modeling complete. Model saved to 'models/logistic_regression_fraud_detection_model'")
        return auc



if __name__ == "__main__":
    preprocessed_train_path = "data/preprocessed/train"
    preprocessed_test_path = "data/preprocessed/test"
    model_output_path = "models/logistic_regression_fraud_detection_model"
    main_modeling(preprocessed_train_path, preprocessed_test_path, model_output_path)
