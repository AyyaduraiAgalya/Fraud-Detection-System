from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import os
import matplotlib.pyplot as plt

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
    lr = LogisticRegression(featuresCol="scaled_features", labelCol="Class", predictionCol="prediction")
    pipeline = Pipeline(stages=[lr])
    model = pipeline.fit(train_df)
    return model

def evaluate_model(model, test_df):
    """
    Evaluates the model on the test set with additional metrics.
    """
    predictions = model.transform(test_df)

    # Basic AUC Evaluation
    evaluator = BinaryClassificationEvaluator(labelCol="Class", metricName="areaUnderROC")
    auc_score = evaluator.evaluate(predictions)
    print(f"Test AUC: {auc_score:.4f}")

    # Converting predictions to RDD format for MulticlassMetrics
    predictions_and_labels = predictions.select("prediction", "Class").rdd.map(lambda x: (float(x[0]), float(x[1])))
    metrics = MulticlassMetrics(predictions_and_labels)

    # Calculating and printing advanced metrics
    precision = metrics.precision(1.0)
    recall = metrics.recall(1.0)
    f1 = metrics.fMeasure(1.0)
    fp_rate = metrics.falsePositiveRate(1.0)
    tp_rate = metrics.truePositiveRate(1.0)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"False Positive Rate: {fp_rate:.4f}")
    print(f"True Positive Rate: {tp_rate:.4f}")

    # Plotting Precision-Recall and ROC Curves
    plot_precision_recall_roc(predictions)

    return auc_score

def plot_precision_recall_roc(predictions):
    """
    Plots Precision-Recall and ROC curves using sklearn.
    """
    # Collecting true labels and prediction probabilities
    y_true = [row["Class"] for row in predictions.select("Class").collect()]
    y_scores = [row["probability"][1] for row in predictions.select("probability").collect()]

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(12, 5))

    # Precision-Recall Curve
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, marker='.')
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    # ROC Curve
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, marker='.')
    plt.title(f"ROC Curve (AUC = {auc_score:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.show()

def main_modeling(train_path: str, test_path: str, model_output_path: str, use_cache=True):
    """
    Main modeling function to train the model, evaluate, and save output.
    """
    if use_cache and os.path.exists(model_output_path):
        print("Model already exists. Skipping modeling step.")
        return
    else:
        # Load preprocessed data
        train_df, test_df = load_preprocessed_data(train_path, test_path)

        # Train model
        model = train_logistic_regression(train_df)

        # Evaluate model with advanced metrics
        auc_score = evaluate_model(model, test_df)
        print(f"Model training complete with AUC: {auc_score:.4f}")

        # Save the model
        model.write().overwrite().save(model_output_path)
        print(f"Model saved to '{model_output_path}'")

if __name__ == "__main__":
    preprocessed_train_path = "data/preprocessed/train"
    preprocessed_test_path = "data/preprocessed/test"
    model_output_path = "models/logistic_regression_fraud_detection_model"
    main_modeling(preprocessed_train_path, preprocessed_test_path, model_output_path)
