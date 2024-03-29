import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify, render_template

# Load the pre-trained model and tokenizer
model_name = "AllyArc/llama_allyarc"  # Replace with your desired model name
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    do_sample=True,
    temperature=0.9,
    top_p=0.6
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize Flask app
app = Flask(__name__)

# Chatbot logic
def generate_response(input_text):
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Create attention mask
    attention_mask = torch.ones_like(input_ids)

    # Generate the response
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=100,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,  # Set do_sample to True
        temperature=0.9,  # Adjust the temperature if needed
        top_p=0.6  # Adjust the top_p if needed
    )

    # Decode the generated response
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

# Serve the HTML file
@app.route('/')
def home():
    return render_template('index.html')

# API endpoint for chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    # Get the user input from the request
    user_input = request.json['input']

    # Generate the response
    response = generate_response(user_input)

    # Return the response as JSON
    return jsonify({'response': response})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
