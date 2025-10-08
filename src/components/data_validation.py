# Import necessary modules
from src.logger import logging
from src.exception import CustomException
import sys
import pandas as pd

# -----------------------------
# Data Validation Class
# -----------------------------
class DataValidation:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def initiate_data_validation(self):
        try:
            logging.info("Starting data validation")

            # 1. Read CSV files
            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)
            logging.info("Train and test data loaded successfully")

            # 2. Check for missing values
            if train_df.isnull().sum().any():
                logging.warning(f"Missing values in training data:\n{train_df.isnull().sum()}")
            else:
                logging.info("No missing values in training data")

            if test_df.isnull().sum().any():
                logging.warning(f"Missing values in testing data:\n{test_df.isnull().sum()}")
            else:
                logging.info("No missing values in testing data")

            # 3. Check required columns
            required_columns = [
                'Age','BusinessTravel','DailyRate','Department','DistanceFromHome',
                'Education','EducationField','EmployeeCount','EmployeeNumber','EnvironmentSatisfaction',
                'Gender','HourlyRate','JobInvolvement','JobLevel','JobRole','JobSatisfaction',
                'MaritalStatus','MonthlyIncome','NumCompaniesWorked','OverTime','PercentSalaryHike',
                'PerformanceRating','RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears',
                'TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole',
                'YearsSinceLastPromotion','YearsWithCurrManager'
            ]

            missing_train_cols = [col for col in required_columns if col not in train_df.columns]
            missing_test_cols = [col for col in required_columns if col not in test_df.columns]

            if missing_train_cols:
                logging.warning(f"Missing columns in training data: {missing_train_cols}")
            else:
                logging.info("All required columns are present in training data")

            if missing_test_cols:
                logging.warning(f"Missing columns in testing data: {missing_test_cols}")
            else:
                logging.info("All required columns are present in testing data")

            logging.info("Data validation completed successfully")
            return True  # Can return True if validation passes

        except Exception as e:
            logging.error("Error occurred during data validation")
            raise CustomException(e, sys)
