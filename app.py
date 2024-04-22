from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('./ml_model/trained_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    input_data = request.json

    # Perform prediction using the loaded model
    prediction = model.predict([[
        input_data['temp'],
        input_data['pulse'],
        input_data['oxygen_saturation']
    ]])[0]

    # Return prediction as JSON response
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    # Run the app with Gunicorn as the WSGI server
    app.run(debug=True, host='0.0.0.0', port=5000)
