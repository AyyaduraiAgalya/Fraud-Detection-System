
# Credit Card Fraud Detection using Distributed Processing and Machine Learning

## Project Overview
This project uses the **Credit Card Fraud Detection dataset** from Kaggle to build a machine learning model that can effectively identify fraudulent transactions. Using **PySpark** for distributed processing, to simulate scalability and real-world data handling, even with a manageable dataset size. This project demonstrates end-to-end data science skills, including data ingestion, exploratory data analysis (EDA), feature engineering, and model development, all optimised for large-scale processing.

## Objectives
1. **Fraud Detection**: Build a predictive model to classify transactions as fraudulent or legitimate.
2. **Scalability with Distributed Processing**: Use PySpark to handle data processing and feature engineering, demonstrating proficiency in scalable data science workflows.
3. **Feature Engineering for Fraud Detection**: Develop fraud-related features to improve model performance and capture potential fraud patterns.

## Project Structure

- `README.md`: Project documentation.
- `data/`: Folder containing the original dataset (not included in the GitHub repository; download instructions provided below).
- `notebooks/`: Jupyter notebooks for initial data exploration and development.
- `scripts/`: Python scripts for modularised code, including data loading, cleaning, and feature engineering.

## Requirements
- **Python 3.6+**
- **PySpark**: Distributed processing for data loading, cleaning, and feature engineering.
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
   - **Script**: `scripts/load_data.py`

2. **Exploratory Data Analysis (EDA)**
   - **Objective**: Understand the data distribution, check for missing values, and analyze class imbalance.
   - **Key Steps**:
     - Generate summary statistics to explore numerical columns.
     - Calculate and visualize the distribution of fraud (Class column) vs. non-fraud transactions.
   - **Outcome**: Insights into the data’s characteristics and imbalance, guiding the model-building stage.
   - **Script**: `scripts/eda.py`

3. **Feature Engineering**
   - **Objective**: Engineer features relevant to fraud detection, such as transaction timing and high-value transaction flags.
   - **Key Features**:
     - `Transaction_Hour`: Extracted from the `Time` column to observe trends based on the hour of the transaction.
     - `High_Amount`: Flags high-value transactions that may indicate fraud.
   - **Tool**: PySpark for efficient distributed feature creation.
   - **Script**: `scripts/feature_engineering.py`

## How to Use

1. **Download the Dataset**
   - Download the **Credit Card Fraud Detection dataset** from Kaggle and save it in the `data/` folder.
   - [Link to dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

2. **Run the Scripts**
   - **Data Loading**: Load data using PySpark by running `load_data.py`.
   - **EDA**: Run `eda.py` to perform exploratory data analysis.
   - **Feature Engineering**: Execute `feature_engineering.py` to generate additional features.

   Example command:
   ```bash
   spark-submit scripts/load_data.py
   ```

3. **Next Steps**
   - **Model Building**: Build and evaluate a machine learning model to detect fraudulent transactions.
   - **Advanced Feature Engineering**: Further enhance features to improve model accuracy.
   - **Evaluation Metrics**: Establish and track metrics like detection rate and alert rate for model performance.

## Results (to be updated)
- Summary of model performance metrics (to be added after model training).
- Success metrics such as **detection rate** and **alert rate** to demonstrate model efficacy in identifying fraud.

## Skills Highlighted
- **Distributed Processing**: Using PySpark to process and engineer features in a scalable environment.
- **Data Analysis and Feature Engineering**: Applying techniques to improve fraud detection model performance.
- **Python and PySpark**: Working with industry-standard tools for data science in financial fraud detection.

## Future Enhancements
- **Integration of Distributed Machine Learning**: Extend the project by applying PySpark MLlib for distributed model training.
- **Dashboard Visualization**: Develop a Tableau dashboard to visualise fraud detection insights.
- **A/B Testing**: Incorporate A/B testing frameworks to test model performance in real-world scenarios.

---
