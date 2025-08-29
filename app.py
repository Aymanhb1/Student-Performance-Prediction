from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None
training_columns = None

def load_model_and_scaler():
    """Load the trained model, scaler, and training columns"""
    global model, scaler, training_columns
    
    # Load model
    MODEL_PATH = 'best_student_performance_model.pkl'
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
    else:
        print(f"❌ Model file not found at {MODEL_PATH}")
        return False
    
    # Load scaler
    SCALER_PATH = 'trained_scaler.pkl'
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print("✅ Scaler loaded successfully")
    else:
        print(f"❌ Scaler file not found at {SCALER_PATH}")
        return False
    
    # Load training columns
    COLUMNS_FILE = 'columns.txt'
    if os.path.exists(COLUMNS_FILE):
        with open(COLUMNS_FILE, 'r') as f:
            training_columns = [line.strip() for line in f if line.strip()]
        print(f"✅ Training columns loaded: {len(training_columns)} features")
    else:
        print(f"❌ Columns file not found at {COLUMNS_FILE}")
        return False
    
    return True

@app.route('/')
def index():
    """Main page with the prediction form"""
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None or scaler is None or training_columns is None:
            return jsonify({
                'error': 'Model, scaler, or training columns not loaded properly',
                'success': False
            })
        
        # Get form data
        data = request.json if request.json else request.form.to_dict()
        
        # Initialize input data with zeros for all training columns
        input_data = {col: 0.0 for col in training_columns}
        
        # Map form inputs to model features
        feature_mapping = {
            'age': 'age',
            'studytime': 'studytime', 
            'absences': 'absences',
            'average_grade': 'Average Grade',
            'attendance_ratio': 'Attendance Ratio',
            'g1': 'G1',
            'g2': 'G2', 
            'g3': 'G3',
            'failures': 'failures',
            'medu': 'Medu',
            'schoolsup': 'schoolsup_yes',
            'higher': 'higher_yes',
            'internet': 'internet_yes',
            'famsup': 'famsup_yes'
        }
        
        # Process numerical features
        for form_key, model_feature in feature_mapping.items():
            if form_key in data and model_feature in training_columns:
                if form_key in ['schoolsup', 'higher', 'internet', 'famsup']:
                    # Binary features
                    input_data[model_feature] = 1.0 if data[form_key] == 'yes' else 0.0
                else:
                    # Numerical features
                    input_data[model_feature] = float(data[form_key])
        
        # Create DataFrame with correct column order
        input_df = pd.DataFrame([input_data])[training_columns]
        
        # Scale the input data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Get prediction probabilities if available
        confidence = None
        probabilities = {}
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_scaled)[0]
            confidence = np.max(proba) * 100
            
            # Map probabilities to class names
            if hasattr(model, 'classes_'):
                for i, class_name in enumerate(model.classes_):
                    probabilities[class_name] = float(proba[i])
        
        # Prepare response
        result = {
            'success': True,
            'prediction': prediction,
            'confidence': round(confidence, 1) if confidence else None,
            'probabilities': probabilities,
            'recommendation': get_recommendation(prediction)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        })

def get_recommendation(prediction):
    """Get recommendation based on prediction"""
    recommendations = {
        'High Risk': {
            'message': 'Intervention Recommended',
            'details': 'This student may benefit from additional support and monitoring.',
            'color': 'danger'
        },
        'Very High Risk': {
            'message': 'Immediate Intervention Required', 
            'details': 'This student needs immediate additional support and close monitoring.',
            'color': 'danger'
        },
        'Medium Risk': {
            'message': 'Monitor Progress',
            'details': 'Keep track of this student\'s performance and provide support as needed.',
            'color': 'warning'
        },
        'Low Risk': {
            'message': 'On Track',
            'details': 'Student appears to be performing well academically.',
            'color': 'success'
        },
        'Very Low Risk': {
            'message': 'Excellent Performance',
            'details': 'Student is performing exceptionally well.',
            'color': 'success'
        }
    }
    
    return recommendations.get(prediction, {
        'message': 'Assessment Complete',
        'details': 'Review student performance regularly.',
        'color': 'info'
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'columns_loaded': training_columns is not None,
        'features_count': len(training_columns) if training_columns else 0
    }
    return jsonify(status)

if __name__ == '__main__':
    print("🚀 Starting Flask Student Performance Prediction App...")
    
    # Load model and dependencies
    if load_model_and_scaler():
        print("✅ All components loaded successfully!")
        print("🌐 Starting server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load required components. Please check:")
        print("  - best_student_performance_model.pkl")
        print("  - trained_scaler.pkl") 
        print("  - columns.txt")

app = Flask(__name__, template_folder='.')
