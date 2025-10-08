from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load preprocessor and model
preprocessor_path = "artifacts/preprocessor.pkl"
model_path = "artifacts/best_model.pkl"

with open(preprocessor_path, 'rb') as f:
    preprocessor = pickle.load(f)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load feature names dynamically from preprocessor
feature_names = preprocessor.feature_names_in_  # Works if you used ColumnTransformer or similar

@app.route('/')
def home():
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        form_data = request.form.to_dict()
        df = pd.DataFrame([form_data])
        
        # Preprocess and predict
        input_transformed = preprocessor.transform(df)
        prediction = model.predict(input_transformed)
        pred_label = "Yes" if prediction[0] == 1 else "No"
        
        return render_template('index.html', features=feature_names, prediction_text=f"Will the employee leave? {pred_label}")
    
    except Exception as e:
        return render_template('index.html', features=feature_names, prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
