from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

default_values = {
    'Gender': 'Female',
    'Age': 35,
    'Tenure': 3,
    'Balance': 50000.0,
    'NumOfProducts': 2,
    'HasCrCard': 'Yes',
    'IsActiveMember': 'No',
    'EstimatedSalary': 60000.0
}

def safe_inference(raw_data: dict):
    customer_id = raw_data.get('CustomerID', 'UNKNOWN')
    data = {}
    for k, default in default_values.items():
        val = raw_data.get(k, default)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            data[k] = default
        else:
            data[k] = val

    df = pd.DataFrame([data])

    df['Gender'] = df['Gender'].astype(str).str.strip().str.lower().replace({
        'male': 'Male', 'female': 'Female'
    })
    if df['Gender'].iloc[0] not in ['Male', 'Female']:
        df['Gender'] = default_values['Gender']

    df['HasCrCard'] = df['HasCrCard'].astype(str).str.strip().replace({
        '1.0': 1, '0.0': 0, 'Yes': 1, 'No': 0
    })
    try:
        df['HasCrCard'] = df['HasCrCard'].astype(int)
    except:
        df['HasCrCard'] = int(default_values['HasCrCard'] == 'Yes')

    df['IsActiveMember'] = df['IsActiveMember'].astype(str).str.strip().replace({
        '1.0': 1, '0.0': 0, 'No': 0, 'Yes': 1
    })
    try:
        df['IsActiveMember'] = df['IsActiveMember'].astype(int)
    except:
        df['IsActiveMember'] = int(default_values['IsActiveMember'] == 'Yes')

    df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

    if 'Gender_Male' not in df.columns:
        df['Gender_Male'] = 0

    expected_cols = ['Age', 'Tenure', 'Balance', 'NumOfProducts',
                     'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Gender_Male']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = default_values.get(col, 0)

    df = df[expected_cols]
    df.fillna({col: default_values.get(col, 0) for col in df.columns}, inplace=True)

    try:
        df_scaled = scaler.transform(df)
    except Exception as e:
        return {'error': f'Scaling failed: {e}'}

    prediction = model.predict(df_scaled)
    prediction_proba = model.predict_proba(df_scaled)

    return {
        'CustomerID': customer_id,
        'PredictedChurn': int(prediction[0]),
        'ChurnProbability': round(prediction_proba[0][1], 4)
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = {k: request.form.get(k) for k in request.form}
    result = safe_inference(user_input)
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
