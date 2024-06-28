from flask import Flask, request, jsonify
from flask_httpauth import HTTPBasicAuth
from .llm_api import process_query
from .database import log_interaction, get_last_messages
from .stt_api import convert_audio_to_text
import os
import re
import json
import uuid
import io
import wave
import numpy as np

app = Flask(__name__)
auth = HTTPBasicAuth()

API_PASSWORD = 'lookatmeimanapikey'

@auth.verify_password
def verify_password(username, password):
    return password == API_PASSWORD

@app.route('/api/query', methods=['POST'])
@auth.login_required
def query_api():
    if request.content_type.startswith('multipart/form-data'):
        # Handle multipart form data (audio file)
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        query_text = process_audio(audio_file)
    else:
        # Handle JSON data
        data = request.get_json()
        query_text = data.get('query', '')

    if not query_text:
        return jsonify({'error': 'Query text is missing or audio processing failed'}), 400

    # Retrieve the last X messages from the log
    num_messages = int(os.getenv("MESSAGE_HISTORY", 10))
    message_history = get_last_messages(num_messages)

    # Process the query using llm_api with message history
    response_text = process_query(query_text, message_history)

    # Extract the JSON object from the response using regular expressions
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    response_json = None
    if json_match:
        json_string = json_match.group()
        try:
            response_json = json.loads(json_string)
        except json.JSONDecodeError:
            pass

    short_answer = response_json.get("short_answer") if response_json else None

    # Log the interaction
    query_uuid = str(uuid.uuid4())
    log_interaction(query_uuid, "", query_text, response_text, "")

    return jsonify({'short_answer': short_answer})

def process_audio(audio_file):
    # Save the uploaded PCM file temporarily
    temp_pcm_path = f"./workspace/temp_{uuid.uuid4()}.pcm"
    audio_file.save(temp_pcm_path)

    # Convert PCM to WAV
    temp_wav_path = f"./workspace/temp_{uuid.uuid4()}.wav"
    with open(temp_pcm_path, 'rb') as pcm_file:
        pcm_data = pcm_file.read()

    with wave.open(temp_wav_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(44100)  # 44.1kHz (matching the Android recording)
        wav_file.writeframes(pcm_data)

    # Convert audio to text
    query_text = convert_audio_to_text(temp_wav_path)

    # Remove temporary files
    os.remove(temp_pcm_path)
    os.remove(temp_wav_path)

    return query_text

if __name__ == '__main__':
    app.run(debug=True)
