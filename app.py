import torch
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('text', '')
        if not data:
            return jsonify({'error': 'Text input is required'}), 400

        # ✅ Load model only when required to save memory
        model = pipeline("text-generation",
                         model="distilbert-base-uncased",
                         torch_dtype=torch.float16,  # Use float16 to reduce RAM
                         device=-1)  # ✅ Force CPU to avoid GPU memory errors

        # Generate response
        result = model(data, max_length=50, num_return_sequences=1)
        return jsonify({'response': result[0]['generated_text']})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  # ✅ Get port from environment
    app.run(host="0.0.0.0", port=port)
