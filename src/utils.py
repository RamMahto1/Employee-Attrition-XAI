import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Train models with GridSearchCV, evaluate, and return the best model.
    """
    try:
        best_score = 0
        best_model_name = None
        best_model = None
        report = []

        for model_name, model in models.items():
            logging.info(f"Training {model_name}...")
            param = params.get(model_name, {})
            gs = GridSearchCV(model, param, cv=3, n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)

            # Set best params and fit again
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Metrics
            f1_train = f1_score(y_train, y_train_pred)
            f1_test = f1_score(y_test, y_test_pred)
            acc_test = accuracy_score(y_test, y_test_pred)
            precision_test = precision_score(y_test, y_test_pred)
            recall_test = recall_score(y_test, y_test_pred)
            roc_auc_test = roc_auc_score(y_test, y_test_pred)

            logging.info(f"{model_name} - Test F1 Score: {f1_test:.4f}, Accuracy: {acc_test:.4f}")

            report.append({
                "Model": model_name,
                "Train F1": f1_train,
                "Test F1": f1_test,
                "Test Accuracy": acc_test,
                "Test Precision": precision_test,
                "Test Recall": recall_test,
                "Test ROC AUC": roc_auc_test
            })

            if f1_test > best_score:
                best_score = f1_test
                best_model_name = model_name
                best_model = model

        return report, best_model_name, best_score, best_model

    except Exception as e:
        raise CustomException(e, sys)
