import pickle
import numpy as np
from flask import Flask, request, jsonify

# Load the model and vectorizer
model_file = 'model.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Initialize the Flask App
app = Flask('insurance_prediction')

# Create the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    
    # Predict and reverse the log transformation
    y_pred_log = model.predict(X)[0]
    y_pred_cost = np.expm1(y_pred_log)

    result = {
        'predicted_medical_cost': float(y_pred_cost)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)