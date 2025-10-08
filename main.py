# main.py
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

if __name__ == "__main__":
    logging.info("=== Pipeline Execution Started ===")

    # Step 1: Data Ingestion
    logging.info("Step 1: Data Ingestion")
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()
    logging.info(f"Train CSV: {train_path}")
    logging.info(f"Test CSV: {test_path}")

    # Step 2: Data Validation
    logging.info("Step 2: Data Validation")
    data_validation = DataValidation(train_path, test_path)
    data_validation.initiate_data_validation()
    logging.info("Data Validation Completed Successfully")

    # Step 3: Data Transformation
    logging.info("Step 3: Data Transformation")
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)
    logging.info(f"Train array shape: {train_arr.shape}")
    logging.info(f"Test array shape: {test_arr.shape}")
    logging.info(f"Preprocessor saved at: {preprocessor_path}")

    # Step 4: Model Training & Evaluation
    logging.info("Step 4: Model Training and Evaluation")
    model_trainer = ModelTrainer()
    best_model_name, best_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    logging.info(f"Best Model: {best_model_name} with F1 Score: {best_score:.4f}")

    logging.info("=== Pipeline Execution Completed Successfully ===")
