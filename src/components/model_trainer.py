import sys
import os
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'best_model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            # Split features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Encode target
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            # Handle imbalance using SMOTE
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

            logging.info(f"After SMOTE, target distribution: {dict(zip(*np.unique(y_train_res, return_counts=True)))}")

            # Define models and hyperparameters
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGB Classifier": XGBClassifier(eval_metric='logloss'),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "SVC": SVC()
            }

            params = {
                "Logistic Regression": {'C':[0.1,1,10], 'solver':['liblinear']},
                "Decision Tree": {'criterion':['gini','entropy'], 'max_depth':[3,5,10,None]},
                "Random Forest": {'n_estimators':[50,100], 'max_depth':[3,5,10,None]},
                "Gradient Boosting": {'learning_rate':[0.01,0.1], 'n_estimators':[50,100]},
                "XGB Classifier": {'learning_rate':[0.01,0.1], 'n_estimators':[50,100]},
                "CatBoost Classifier": {'depth':[4,6], 'learning_rate':[0.01,0.1], 'iterations':[50,100]},
                "AdaBoost Classifier": {'learning_rate':[0.01,0.1], 'n_estimators':[50,100]},
                "SVC": {'C':[0.1,1], 'kernel':['linear','rbf']}
            }

            # Evaluate models
            report, best_model_name, best_score, best_model = evaluate_models(
                X_train_res, y_train_res, X_test, y_test, models, params
            )

            logging.info(f"Best model: {best_model_name} with F1 Score: {best_score:.4f}")

            # Save best model
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logging.info("Best model saved successfully.")

            return best_model_name, best_score

        except Exception as e:
            logging.error("Error in Model Trainer")
            raise CustomException(e, sys)
