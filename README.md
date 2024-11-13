
# Credit Card Fraud Detection using Distributed Processing and Machine Learning

## Project Overview
This project uses the **Credit Card Fraud Detection dataset** from Kaggle to build a machine learning model that can effectively identify fraudulent transactions. Using **PySpark** for distributed processing, to simulate scalability and real-world data handling, even with a manageable dataset size. This project demonstrates end-to-end data science skills, including data ingestion, exploratory data analysis (EDA), feature engineering, preprocessing, and model development, all optimised for large-scale processing.

## Objectives
1. **Fraud Detection**: Build a predictive model to classify transactions as fraudulent or legitimate.
2. **Scalability with Distributed Processing**: Use PySpark to handle data processing and feature engineering, demonstrating proficiency in scalable data science workflows.
3. **Feature Engineering for Fraud Detection**: Develop fraud-related features to improve model performance and capture potential fraud patterns.

## Project Structure

- `README.md`: Project documentation.
- `data/`: Folder containing the original dataset (not included in the GitHub repository; download instructions provided below), engineered, and preprocessed data.
- `notebooks/`: Jupyter notebooks for initial data exploration and feature engineering.
- `scripts/`
  - `data_loading.py`: Script for loading raw data into the PySpark environment.
  - `eda.py` : Script for exploratory data analysis.
  - `feature_engineering.py`: Script for engineering time-based and transaction-based features relevant to fraud detection.
  - `preprocessing.py`: Script to handle data preprocessing, including ensuring numerical data types, scaling features, and splitting data into train/test sets.
  - `modelling.py`: Script to train and evaluate the fraud detection model.
  - `main.py`: The main script orchestrating the entire pipeline from data loading, feature engineering, preprocessing, to model training and evaluation.

## Requirements
- **Python 3.6+**
- **PySpark**: Distributed processing for data loading, cleaning, feature engineering, and model training.
- **Pandas, NumPy**: Core libraries for data manipulation.

To install required packages:
```bash
pip install -r requirements.txt
```

## Steps to Reproduce

1. **Data Collection and Loading**
   - **Tool**: PySpark
   - **Description**: Load the dataset using PySpark DataFrames to enable distributed data processing.
   - **Details**: Reads a CSV file and infers schema automatically for a smooth transition into EDA.
   - **Script**: `scripts/data_loading.py`

2. **Exploratory Data Analysis (EDA)**
   - **Objective**: Understand the data distribution, check for missing values, and analyse class imbalance.
   - **Outcome**: Insights into the data’s characteristics and imbalance, guiding the model-building stage.
   - **Script**: `scripts/eda.py`

3. **Feature Engineering**
   - **Objective**: Engineer features that highlight transaction frequency, average amount, and value patterns commonly associated with fraud.
   - **Key Features**:
     - **Transactions_Last_1_Hour** and **Transactions_Last_5_Mins**: Counts of transactions within the last 1-hour and 5-mins windows to detect high transaction frequency, a common fraud pattern.
     - **Avg_Amount_Last_1_Hour** and **Avg_Amount_Last_5_Mins**: The average transaction amount within the last 1-hour and 5-mins window, helping to identify sudden spending increases.
     - **High_Amount**: Flags high-value transactions that exceed a predefined threshold (e.g., $2000), which can signal abnormal spending behavior.
     - **Stddev_Amount_Last_1_Hour** and **Stddev_Amount_Last_5_Mins**: The rolling standard deviation of transaction amounts within the last 1-hour and 5-mins window, capturing transaction variability.
   - **Tool**: PySpark for efficient distributed feature creation on large datasets.
   - **Script**: `scripts/feature_engineering.py`

4. **Preprocessing**
   - **Objective**: Prepare data for modeling by scaling numerical features and splitting data into training and testing sets.
   - **Processes**:
     - **Scaling**: Standardise numerical features to ensure uniform data distributions.
     - **Train-Test Split**: Split data into training and testing sets for model evaluation.
   - **Script**: `scripts/preprocessing.py`

5. **Modelling**
   - **Objective**: Train and evaluate a machine learning model for fraud detection.
   - **Key Steps**:
     - **Model Training**: Train a logistic regression model with PySpark MLlib.
     - **Evaluation Metrics**: Evaluate model performance using AUC (Area Under the Curve) to measure the model’s accuracy in distinguishing fraudulent from non-fraudulent transactions.
   - **Script**: `scripts/modelling.py`

## How to Use

1. **Download the Dataset**
   - Download the **Credit Card Fraud Detection dataset** from Kaggle and save it in the `data/raw` folder.
   - [Link to dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

2. **Setup the Environment**
   - Ensure that PySpark is installed and properly configured in your environment.
   - Optionally, set up a virtual environment for managing dependencies.

3. **Run the Full Pipeline**
   - Use `main.py` to execute the entire pipeline:
   ```bash
   python scripts/main.py
   ```
   This command will:
   - Load the raw data from the `data/raw` folder.
   - Perform feature engineering using PySpark.
   - Preprocess the engineered data, including scaling and formatting.
   - Train and evaluate the logistic regression model.
   - Save the model and output relevant performance metrics.

4. **Run Individual Scripts**
   - **Data Loading**: Load the dataset as a standalone step if needed:
     ```bash
     spark-submit scripts/data_loading.py
     ```
   - **Exploratory Data Analysis (EDA)**: For initial data insights, run:
     ```bash
     spark-submit scripts/eda.py
     ```
   - **Feature Engineering**: To add features to the dataset, run:
     ```bash
     spark-submit scripts/feature_engineering.py
     ```
   - **Preprocessing**: Apply scaling and transformations to prepare data for modelling:
     ```bash
     spark-submit scripts/preprocessing.py
     ```
   - **Modelling**: Train and evaluate the model independently if needed:
     ```bash
     spark-submit scripts/modelling.py
     ```

## Results
- **Model Performance**: The logistic regression model achieved an AUC score of 0.9709, indicating strong predictive performance in identifying fraudulent transactions.

## Skills Highlighted
- **Distributed Processing**: Using PySpark to process and engineer features in a scalable environment.
- **Data Analysis and Feature Engineering**: Applying techniques to improve fraud detection model performance.
- **Machine Learning**: Building and evaluating logistic regression models for fraud detection.
- **Python and PySpark**: Working with industry-standard tools for data science in financial fraud detection.

## Future Enhancements
- **Integration of Distributed Machine Learning**: Extend the project by applying PySpark MLlib for distributed model training.
- **Dashboard Visualisation**: Develop a Tableau dashboard to visualise fraud detection insights.
- **A/B Testing**: Incorporate A/B testing frameworks to test model performance in real-world scenarios.

## Dataset Acknowledgment
This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), available on Kaggle and provided by the Machine Learning Group at ULB. This dataset is made available under the Database Contents License (DbCL) v1.0, which permits use with attribution.
To comply with the DbCL license, this project does not redistribute the original data. Instead, interested users can download it directly from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---
