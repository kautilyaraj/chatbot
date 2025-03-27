from flask import Flask, request, render_template

app = Flask(__name__)

def get_response(user_input):
    user_input = user_input.lower()
    if "hi" in user_input or "hello" in user_input:
        return "Hello! How can I assist you today?"
    elif "problem" in user_input or "issue" in user_input:
        return "Please tell me more about your problem!"
    elif "bye" in user_input:
        return "Goodbye! Have a great day!"
    else:
        return "Sorry, I didn’t get that. Can you explain more?"

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_message = request.form["message"]
        bot_response = get_response(user_message)
        return render_template("index.html", user_message=user_message, bot_response=bot_response)
    return render_template("index.html")

if __name__ == "__main__":
    app.run()