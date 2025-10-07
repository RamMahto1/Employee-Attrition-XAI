from src.logger import logging
from src.exception import CustomException
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation


## step 1: data ingestion
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()
    logging.info(f"Train data path: {train_data}")
    logging.info(f"Test data path: {test_data}")
    
    # step 2: data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)
    logging.info(f"Train array shape: {train_arr.shape}")
    logging.info(f"Test array shape: {test_arr.shape}")
   

# try:
#     logging.info("Starting main.py execution")
   
# except Exception as e:
#     raise CustomException(e, sys)
# logging.info("logger imported successfully")