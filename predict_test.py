import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import os

# Load the preprocessor and model
preprocessor_path = "artifacts/preprocessor.pkl"
model_path = "artifacts/best_model.pkl"

# Load preprocessor and model
with open(preprocessor_path, 'rb') as f:
    preprocessor = pickle.load(f)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load a few rows from test data
test_data = pd.read_csv("artifacts/test.csv")

# Separate features and target
X_test = test_data.drop("Attrition", axis=1)
y_test = test_data["Attrition"]

# Transform and predict
X_test_transformed = preprocessor.transform(X_test)
predictions = model.predict(X_test_transformed)




# Compare some predictions
results = pd.DataFrame({
    "Actual": y_test.values[:20],
    "Predicted": predictions[:20]
})

print("Model prediction test successful!\n")
print(results)


# Evaluate 
print("\nConfusion Matrix:")
print(confusion_matrix(y_test.map({'No':0, 'Yes':1}), predictions))

print("\nClassification Report:")
print(classification_report(y_test.map({'No':0, 'Yes':1}), predictions, target_names=['No', 'Yes']))



