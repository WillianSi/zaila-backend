from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

HF_API_KEY = os.getenv("HF_API_KEY")

if not HF_API_KEY:
    raise Exception("HF_API_KEY não encontrado! Configure no arquivo .env")

client = InferenceClient(provider="novita", api_key=HF_API_KEY)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3-0324",
        messages=[{"role": "user", "content": user_message}]
    )

    response_text = completion.choices[0].message['content']
    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
