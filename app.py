import os
import torch
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# ✅ Home Route to Test API
@app.route('/')
def home():
    return "Flask API is running! Use the /predict endpoint."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('text', '')
        if not data:
            return jsonify({'error': 'Text input is required'}), 400

        # ✅ Load Model Once to Reduce Memory Usage
        if not hasattr(app, 'model'):
            app.model = pipeline("text-generation", 
                                 model="distilbert-base-uncased", 
                                 device=-1)  # ✅ Force CPU Usage

        result = app.model(data, max_length=50, num_return_sequences=1)
        return jsonify({'response': result[0]['generated_text']})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  
    app.run(host="0.0.0.0", port=port)
