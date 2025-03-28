import torch
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Input text from request
        data = request.json.get('text', '')
        if not data:
            return jsonify({'error': 'Text input is required'}), 400

        # Load model inside function to save memory
        model = pipeline("text-generation", 
                         model="distilbert-base-uncased", 
                         torch_dtype=torch.float16,  # Quantization to reduce RAM
                         device=0 if torch.cuda.is_available() else -1)  # Use GPU if available

        # Generate response
        result = model(data, max_length=50, num_return_sequences=1)
        return jsonify({'response': result[0]['generated_text']})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


import os  # Add this at the top of your file

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Get PORT from environment
    app.run(host="0.0.0.0", port=port)
