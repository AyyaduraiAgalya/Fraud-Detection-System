from feature_engineering import main_feature_engineering
from preprocessing import main_preprocessing


def main():
    raw_data_path = "data/raw/creditcard.csv"
    engineered_data_path = "data/engineered"
    preprocessed_data_path = "data/preprocessed"

    # Step 1: Feature Engineering
    print("Starting feature engineering...")
    main_feature_engineering(raw_data_path, engineered_data_path)
    print("Feature engineering completed.")

    # Step 2: Data Preprocessing
    print("Starting data preprocessing...")
    main_preprocessing(engineered_data_path, preprocessed_data_path)
    print("Data preprocessing completed.")


if __name__ == "__main__":
    main()
