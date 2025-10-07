from src.logger import logging
from src.exception import CustomException
import sys
import os
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from src.utils import save_object

#  Configuration dataclass
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            numerical_features = [
                'Age','DailyRate','DistanceFromHome','Education','EnvironmentSatisfaction',
                'HourlyRate','JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome',
                'MonthlyRate','NumCompaniesWorked','PercentSalaryHike','PerformanceRating',
                'RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears',
                'TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole',
                'YearsSinceLastPromotion','YearsWithCurrManager'
            ]

            categorical_features = [
                'BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime'
            ]

            # Pipelines
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

            #  Combine
            preprocessor = ColumnTransformer(transformers=[
                ('num', num_pipeline, numerical_features),
                ('cat', cat_pipeline, categorical_features)
            ])

            logging.info("Data transformation pipeline created successfully.")
            return preprocessor

        except Exception as e:
            logging.error("Error in get_data_transformer_object.")
            raise CustomException(e, sys)
      
    def initiate_data_transformation(self, train_path, test_path):
        try:
            #  Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data read successfully.")

            target_column_name = 'Attrition'

            #  Get preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            #  Separate features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            #  Transform the data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            #  Convert targets to numpy
            y_train_arr = np.array(target_feature_train_df)
            y_test_arr = np.array(target_feature_test_df)

            #  Combine transformed input + target
            train_arr = np.c_[input_feature_train_arr, y_train_arr]
            test_arr = np.c_[input_feature_test_arr, y_test_arr]

            #  Save preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor object saved successfully.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error("Error in initiate_data_transformation.")
            raise CustomException(e, sys)
