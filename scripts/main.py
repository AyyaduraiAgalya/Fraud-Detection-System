from pyspark.sql import SparkSession
from feature_engineering import main_feature_engineering
from preprocessing import main_preprocessing
from modelling import main_modeling  # Assuming this exists in modelling.py


def main():
    # Defining paths
    raw_data_path = "data/raw/creditcard.csv"
    engineered_data_path = "data/engineered/engineered_data"
    preprocessed_data_path = "data/preprocessed/preprocessed_data"
    model_output_path = "models/logistic_regression_fraud_detection_model"

    # Step 1: Feature Engineering
    print("Applying feature engineering...")
    main_feature_engineering(raw_data_path, engineered_data_path, use_cache=True)

    # Step 2: Preprocessing
    print("Applying preprocessing...")
    main_preprocessing(engineered_data_path, preprocessed_data_path, use_cache=True)

    # Step 3: Modeling
    print("Training and evaluating model...")
    auc = main_modeling(preprocessed_data_path + "/train", preprocessed_data_path + "/test",
                                   model_output_path, use_cache=False)

    # Only print the AUC if auc is not None (i.e., model was trained)
    if auc is not None:
        print(f"Model training complete with AUC: {auc:.4f}")

if __name__ == "__main__":
    main()