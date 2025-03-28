from flask import Flask, request, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Hugging Face se model aur tokenizer load karna
model_name = "microsoft/DialoGPT-small"  # Hinglish ke liye ek aur model lena padega
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate response
def get_response(user_input):
    user_input = user_input.lower()
    
    # Agar user ka input common hai, toh predefined response do
    if "hi" in user_input or "hello" in user_input:
        return "Hello! Kaise ho?"
    elif "kaisa hai" in user_input:
        return "Sab badhiya! Tu bata?"
    elif "bye" in user_input:
        return "Bye bhai, fir milenge!"
    
    # Hugging Face ka model use karke response generate karna
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    bot_output = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(bot_output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return response

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_message = request.form["message"]
        bot_response = get_response(user_message)
        return render_template("index.html", user_message=user_message, bot_response=bot_response)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
