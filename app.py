import torch
import os
from flask import Flask, request, jsonify
from flask_cors import CORS  # Enable CORS for API calls
from transformers import pipeline

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load Hugging Face model globally
model = pipeline(
    "text-generation", 
    model="distilgpt2",  # Change this to your fine-tuned model
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  
    device=0 if torch.cuda.is_available() else -1
)

@app.route('/')
def home():
    return "Flask API is running! Use the /predict endpoint.", 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Invalid input, expected JSON with 'text' key"}), 400

        input_text = data["text"]
        result = model(input_text, max_length=50, num_return_sequences=1)

        return jsonify({"response": result[0]["generated_text"]}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
