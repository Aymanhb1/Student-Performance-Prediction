import flask
from flask import render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

app = flask.Flask(__name__)

MODEL_PATH = 'best_student_performance_model.pkl'
COLUMNS_FILE = 'columns.txt'

# Load model and columns (your existing logic is fine; simplified here)
with open(MODEL_PATH, 'rb') as f:
    model = joblib.load(f)

with open(COLUMNS_FILE, 'r') as f:
    training_columns = [line.strip() for line in f if line.strip()]

# Dummy scaler to keep your current approach (ideally save the real scaler)
scaler = MinMaxScaler().fit(pd.DataFrame(np.zeros((1, len(training_columns))), columns=training_columns))

@app.route('/', methods=['GET'])
def home():
    # render template with no result initially
    return render_template('form.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1) Read form inputs
        data = request.form.to_dict()

        # 2) To numeric where possible (strings -> numbers)
        for k, v in data.items():
            if isinstance(v, str) and v.strip() != '':
                try:
                    if '.' in v:
                        data[k] = float(v)
                    else:
                        data[k] = int(v)
                except:
                    # leave as string if not numeric
                    pass

        # 3) Align with training columns (missing -> 0)
        aligned = pd.DataFrame(np.zeros((1, len(training_columns))), columns=training_columns)

        for col in data.keys():
            if col in aligned.columns:
                aligned[col] = data[col]  # assign provided values

        # 4) Scale then predict
        X = scaler.transform(aligned)
        y_pred = model.predict(X)
        predicted_risk_category = str(y_pred[0])

        # 5) Re-render the same page with result clearly shown under the table
        return render_template('form.html', result=predicted_risk_category)

    except Exception as e:
        # Show error nicely on the page (optional)
        return render_template('form.html', result=f"Error: {e}")

if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
