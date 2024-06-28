Output of tree command:
```
|-- .DS_Store
|-- README.md
|-- client
|-- docs
    |-- conversate
        |-- api_doc.md
        |-- spec0.0.1.md
    |-- thirdparty
        |-- anthropic.md
        |-- llm_api.md
        |-- lmstudio_api.md
        |-- oi.md
        |-- tts.md
        |-- whisperx.md
|-- entities
    |-- self
        |-- mind_models.yaml
        |-- prompts
            |-- actions.md
            |-- default.md
            |-- dream.md
            |-- journal.md
            |-- response.json
            |-- subconcious
                |-- ekman
                    |-- anger.yaml
                    |-- disgust.yaml
                    |-- fear.yaml
                    |-- joy.yaml
                    |-- sadness.yaml
                |-- plutchik
                    |-- anger.yaml
                    |-- anticipation.yaml
                    |-- disgust.yaml
                    |-- fear.yaml
                    |-- joy.yaml
                    |-- sadness.yaml
                    |-- suprise.yaml
                    |-- trust.yaml
                |-- theories.md
            |-- summarize.md
            |-- summarize_day.md
        |-- self.yaml
        |-- voices
            |-- brimley1.wav
            |-- max1
                |-- fillers
                    |-- checking_on_that.wav
                    |-- give_me_just_a_moment.wav
                    |-- hmm_one_moment.wav
                    |-- let_me_see_here.wav
                    |-- one_moment_please_sam.wav
            |-- max1.wav
            |-- toaster.wav
    |-- user
        |-- user.yaml
|-- env_example
|-- main.py
|-- modules
    |-- __init__.py
    |-- __pycache__
    |-- api.py
    |-- conversate_app.py
    |-- database.py
    |-- llm_api.py
    |-- stt_api.py
    |-- tts_api.py
|-- plugins
|-- requirements.txt
|-- resources
    |-- gfx
    |-- sfx
|-- test_api.py
|-- voices
    |-- max1
        |-- common
|-- workspace

```

---

./modules/conversate_app.py
```
import os
from dotenv import load_dotenv
import time
import uuid
import io
import re
import json

import pygame
import pygame.mixer as mixer
import sounddevice as sd
import soundfile as sf
import numpy as np
import scipy.io.wavfile as wavfile
from threading import Thread

from .llm_api import process_query
from .tts_api import convert_to_speech, tts_available
from .stt_api import convert_audio_to_text
from .database import log_interaction, get_last_messages

load_dotenv()

mixer.init()

class ConversateApp:
    def __init__(self, screen):
        self.screen = screen
        self.screen_width, self.screen_height = screen.get_size()
        self.background_color = (230, 230, 230)
        self.idle_color = (0, 0, 255)
        self.listening_color = (255, 165, 0)
        self.thinking_color = (128, 0, 128)
        self.speaking_color = (0, 255, 0)

        self.sample_rate = 48000
        self.channels = 1

        self.input_device = None
        self.output_device = None
        
        self.typing_mode = False
        self.input_text = ""
        self.font = pygame.font.Font(None, 36)

        self.recording = False
        self.recorded_frames = []

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and not self.typing_mode:
                        # Start recording
                        self.recording = True
                        self.recorded_frames = []
                    elif event.key == pygame.K_RETURN:
                        if not self.typing_mode:
                            self.typing_mode = True
                            self.input_text = ""
                        else:
                            self.typing_mode = False
                            query_text = self.input_text.strip()
                            if query_text:
                                self.process_query(query_text)
                    elif self.typing_mode:
                        if event.key == pygame.K_BACKSPACE:
                            self.input_text = self.input_text[:-1]
                        else:
                            self.input_text += event.unicode
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE and not self.typing_mode:
                        # Stop recording and process the query
                        self.recording = False
                        pygame.draw.circle(self.screen, self.thinking_color, (self.screen_width // 2, self.screen_height // 2), 100)
                        pygame.display.flip()
                        query_uuid = str(uuid.uuid4())
                        query_audio_file = f"./workspace/queries/{query_uuid}.wav"
                        recording = np.concatenate(self.recorded_frames, axis=0)
                        wavfile.write(query_audio_file, self.sample_rate, recording)
                        query_text = convert_audio_to_text(query_audio_file)
                        if query_text == "":
                            print("I didn't hear you.")
                            continue
                        else:
                            print(f"You said: {query_text}")
                        self.process_query(query_text, query_audio_file, query_uuid)
                            
            # Draw the idle circle and text input box
            self.screen.fill(self.background_color)
            if self.recording:
                circle_color = self.listening_color
            else:
                circle_color = self.idle_color
            pygame.draw.circle(self.screen, circle_color, (self.screen_width // 2, self.screen_height // 2), 100)
            
            if self.typing_mode:
                input_surface = self.font.render(self.input_text, True, (0, 0, 0))
                input_rect = input_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
                self.screen.blit(input_surface, input_rect)
            
            pygame.display.flip()

            # Record audio while the spacebar is held down
            if self.recording:
                frame = sd.rec(1024, samplerate=self.sample_rate, channels=self.channels, device=self.input_device)
                sd.wait()
                self.recorded_frames.append(frame)
    
    # Modify the process_query method
    def process_query(self, query_text, query_audio_file="", query_uuid=None):
        start_time = time.time()
        if not query_uuid:
            query_uuid = str(uuid.uuid4())
        
        num_messages = int(os.getenv("MESSAGE_HISTORY", 10))
        message_history = get_last_messages(num_messages)
        
        pygame.draw.circle(self.screen, self.thinking_color, (self.screen_width // 2, self.screen_height // 2), 100)
        pygame.display.flip()
        inference_start_time = time.time()
        response_text = process_query(query_text, message_history)

        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        response_json = None
        if json_match:
            json_string = json_match.group()
            try:
                response_json = json.loads(json_string)
                if "short_answer" in response_json:
                    short_answer = response_json["short_answer"]
                else:
                    print("'short_answer' key not found in the JSON response. Falling back to full answer!")
            except json.JSONDecodeError:
                print("Invalid JSON format. Falling back to full answer!")

        inference_time = time.time() - inference_start_time
        print(f"Inference ... {inference_time} seconds")
        
        # Check if TTS is available before processing TTS
        if tts_available:
            pygame.draw.circle(self.screen, self.speaking_color, (self.screen_width // 2, self.screen_height // 2), 100)
            pygame.display.flip()
            tts_start_time = time.time()
            response_audio_file = f"./workspace/responses/{query_uuid}.wav"
            
            if response_json and "short_answer" in response_json:
                concatenated_audio, sample_rate = convert_to_speech(response_json["short_answer"], query_uuid, self.speak_audio)
            else:
                concatenated_audio, sample_rate = convert_to_speech(response_text, query_uuid, self.speak_audio)

            if concatenated_audio is not None and sample_rate is not None:
                sf.write(response_audio_file, concatenated_audio, sample_rate)
            tts_time = time.time() - tts_start_time
            print(f"TTS ... {tts_time} seconds")

        total_time = time.time() - start_time
        print(f"Turn completed in {total_time} seconds")

        log_interaction(query_uuid, query_audio_file, query_text, response_text, response_audio_file)


    def speak_audio(self, audio_data, sample_rate):
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio_data, sample_rate, format='wav')
        audio_bytes.seek(0)
        sound = mixer.Sound(audio_bytes)
        channel = sound.play()
        # Wait for the playback to finish
        while mixer.get_busy():
            pygame.time.delay(100)

```
---

./modules/stt_api.py
```
import whisperx
import os
from dotenv import load_dotenv
import uuid
import numpy as np

load_dotenv()

def convert_audio_to_text(audio_file):
    model = whisperx.load_model(os.getenv("STT_MODEL"), os.getenv("STT_DEVICE"), compute_type=os.getenv("STT_COMPUTE_TYPE"))

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, language="en", batch_size=16)
    text = ""
    for segment in result["segments"]:
        text += segment["text"]
    return text```
---

./modules/llm_api.py
```
import os
from dotenv import load_dotenv
import json
import yaml
import tiktoken
import re
from .database import load_recent_summaries

load_dotenv()

def load_file_contents(file_path):
    with open(file_path, "r") as file:
        return file.read()

def process_query(query, message_history, is_journal_update=False, config_name="deep_reason"):
    # Load the configuration file
    with open("./entities/self/mind_models.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Get the specified configuration
    client_config = config[config_name]

    # Load the client based on the provider
    if client_config["provider"] == "openai":
        from openai import OpenAI
        client = OpenAI(base_url=os.getenv("LLM_API_URL"), api_key=os.getenv("LLM_API_KEY"))
        print("Going local")
    elif client_config["provider"] == "groq":
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        print("Requesting Groq")
    elif client_config["provider"] == "anthropic":
        import anthropic
        client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))
        print("Requesting Anthropic")
    else:
        raise ValueError(f"Unsupported provider: {client_config['provider']}")

    num_recent_summaries = 3  # Adjust the number of recent summaries to include
    recent_summaries = load_recent_summaries(num_recent_summaries)

    system_prompt = load_file_contents("./entities/self/prompts/default.md").format(
        self_yaml=load_file_contents("./entities/self/self.yaml"),
        user_yaml=load_file_contents("./entities/user/user.yaml"),
        response_json=load_file_contents("./entities/self/prompts/response.json"),
        actions_md=load_file_contents("./entities/self/prompts/actions.md"),
        recent_memories="\n".join(recent_summaries),
    )

    messages = []

    if is_journal_update:
        messages = [{"role": "user", "content": message[0] + "\n" + message[1]} for message in message_history]
        messages.append({"role": "user", "content": query})
    else:
        # Add historical messages to the beginning of the messages list
        for query_text, response_text in message_history:
            if query_text.strip():
                messages.append({"role": "user", "content": query_text})
            if response_text.strip():
                messages.append({"role": "assistant", "content": response_text})

        # Append the current query to the messages list
        if query.strip():
            messages.append({"role": "user", "content": query})

    # get the token count of our system prompt
    print(f"System prompt tokens: {num_tokens_from_string(system_prompt)}")

    max_retries = 3
    retry_count = 0
    response_content = None

    while retry_count < max_retries:
        if client_config["provider"] == "anthropic":
            completion = client.messages.create(
                model=client_config["model"],
                system=system_prompt,
                messages=messages,
                temperature=client_config["temperature"],
                max_tokens=client_config["max_tokens"],
            )
            response_text = completion.content
            if isinstance(response_text, list) and len(response_text) > 0 and hasattr(response_text[0], "text"):
                response_content = response_text[0].text
            else:
                response_content = str(response_text)

        else:
            completion = client.chat.completions.create(
                model=client_config["model"],
                messages=messages,
                temperature=client_config["temperature"],
                max_tokens=client_config["max_tokens"],
            )
            response_text = completion.choices[0].message
            if isinstance(response_text, dict) and "content" in response_text:
                response_content = response_text["content"]
            elif hasattr(response_text, "content"):
                response_content = response_text.content
            else:
                response_content = str(response_text)

        response_json = None
        if response_content and isinstance(response_content, str):
            json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
            if json_match:
                json_string = json_match.group()
                
                try:
                    response_json = json.loads(json_string)
                    if "short_answer" in response_json:
                        short_answer = response_json["short_answer"]
                    else:
                        print("'short_answer' key not found in the JSON response. Falling back to full answer!")
                except json.JSONDecodeError:
                    print("Invalid JSON format. Falling back to full answer!")
        else:
            print("Empty or non-string response received. Skipping JSON parsing.")

        if response_json and all(key in response_json for key in ["internal_thought", "short_answer", "action"]):
            # Serialize the JSON object with escaped newlines
            response_content = json.dumps(response_json)
            break  # Exit the loop if the response matches the expected schema
        else:
            retry_count += 1
            print(f"Retry {retry_count}: Response does not match the expected schema.")

    print(f"\n\nResponse:\n\n{response_content}\n\n")

    return response_content

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens```
---

./modules/database.py
```
import sqlite3
from datetime import datetime
# from .llm_api import process_query
import os
import json

os.environ['TZ'] = 'MST'

def get_db_connection():
    return sqlite3.connect("./workspace/history.db")

def create_interactions_table():
    db_connection = get_db_connection()
    db_cursor = db_connection.cursor()
    db_cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT (datetime('now', 'localtime')),
            query_uuid TEXT,
            query_audio_file TEXT,
            query_text TEXT,
            response_text TEXT,
            response_audio_file TEXT
        )
    """)
    db_connection.commit()
    db_connection.close()

def log_interaction(query_uuid, query_audio_file, query_text, response_text, response_audio_file):
    # print(f"{query_uuid} logged.")
    db_connection = get_db_connection()
    db_cursor = db_connection.cursor()
    db_cursor.execute("""
        INSERT INTO interactions (query_uuid, query_audio_file, query_text, response_text, response_audio_file)
        VALUES (?, ?, ?, ?, ?)
    """, (query_uuid, query_audio_file, query_text, response_text, response_audio_file))
    db_connection.commit()
    db_connection.close()

def get_last_messages(num_messages):
    db_connection = get_db_connection()
    db_cursor = db_connection.cursor()
    try:
        today = datetime.now().date()
        db_cursor.execute("""
            SELECT query_text, response_text
            FROM interactions
            WHERE DATE(timestamp) = ?
            ORDER BY rowid ASC
            LIMIT ?
        """, (today, num_messages))
    except sqlite3.OperationalError:
        print("Could not fetch messages from database.")
        return []
    messages = db_cursor.fetchall()
    db_connection.close()
    
    if not messages:
        return []
    
    return messages

def get_distinct_dates():
    db_connection = get_db_connection()
    db_cursor = db_connection.cursor()
    db_cursor.execute("""
        SELECT DISTINCT DATE(timestamp) AS date
        FROM interactions
        ORDER BY date
    """)
    dates = db_cursor.fetchall()
    db_connection.close()
    return [date[0] for date in dates]

def get_messages_by_date(date):
    db_connection = get_db_connection()
    db_cursor = db_connection.cursor()
    db_cursor.execute("""
        SELECT query_text, response_text
        FROM interactions
        WHERE DATE(timestamp) = ?
        ORDER BY rowid ASC
    """, (date,))
    messages = db_cursor.fetchall()
    db_connection.close()
    return messages

def check_missing_journal_entries(process_query):
    journal_folder = "./workspace/journal"
    summary_folder = "./workspace/summaries"
    os.makedirs(journal_folder, exist_ok=True)
    os.makedirs(summary_folder, exist_ok=True)

    distinct_dates = get_distinct_dates()
    today = datetime.now().date()

    for date in distinct_dates:
        if date == today:
            continue

        journal_file = os.path.join(journal_folder, f"{date}.md")
        summary_file = os.path.join(summary_folder, f"{date}.md")

        if not os.path.exists(journal_file):
            print(f"Journaling for {date}...")
            messages = get_messages_by_date(date)
            journal_prompt = load_file_contents("./entities/self/prompts/journal.md")
            journal_entry = process_query(journal_prompt, messages) 

            # if journal_entry can be loaded as a json object, get short_answer
            try:
                journal_entry = json.loads(journal_entry)
                if "short_answer" in journal_entry:
                    journal_entry = journal_entry["short_answer"]
            except json.JSONDecodeError:
                journal_entry = journal_entry

            with open(journal_file, "w") as file:
                file.write(journal_entry)

            summary_file = os.path.join(summary_folder, f"{date}.md")
        if not os.path.exists(summary_file):
            print(f"Summarizing for {date}...")
            messages = get_messages_by_date(date)
            summary_prompt = load_file_contents("./entities/self/prompts/summarize_day.md")
            summary_entry = process_query(summary_prompt, messages)  # Use the passed process_query function
            with open(summary_file, "w") as file:
                file.write(summary_entry)
                
        print("Journals up to date!")

def load_recent_summaries(num_summaries):
    summary_folder = "./workspace/summaries"
    summary_files = sorted(os.listdir(summary_folder), reverse=True)[:num_summaries]
    summaries = []
    for summary_file in summary_files:
        with open(os.path.join(summary_folder, summary_file), "r") as file:
            summaries.append(file.read())
    return summaries


            

def load_file_contents(filename):
    with open(filename, "r") as file:
        return file.read()
    ```
---

./modules/tts_api.py
```
import gradio_client
import os
from dotenv import load_dotenv
import librosa
import soundfile as sf
import io
import nltk
from queue import Queue
from threading import Thread
import numpy as np
import nltk

load_dotenv()

nltk.download('punkt')

# Global flag to indicate TTS availability
tts_available = True

try: 
    tts_client = gradio_client.Client(os.getenv("TTS_API_URL"))
except Exception as e:
    print(f"Error connecting to TTS API. Make sure xtts2_ui is running: {e}")
    tts_available = False

def convert_to_speech(text, query_uuid, speak_callback):
    if not tts_available:
        print("TTS is unavailable. Skipping TTS conversion.")
        return None, None
    sentences = nltk.sent_tokenize(text)
    audio_queue = Queue()
    concatenation_queue = Queue()
    generation_thread = Thread(target=generate_audio, args=(sentences, query_uuid, audio_queue, concatenation_queue))
    generation_thread.start()
    playback_thread = Thread(target=playback_audio, args=(audio_queue, speak_callback))
    playback_thread.start()

    generation_thread.join()
    audio_queue.put(None)
    playback_thread.join()
    audio_data_list = []
    sample_rate = None
    while not concatenation_queue.empty():
        audio_data = concatenation_queue.get()
        if audio_data is not None:
            audio_data_list.append(audio_data[0])
            if sample_rate is None:
                sample_rate = audio_data[1]
    if len(audio_data_list) > 0:
        concatenated_audio = np.concatenate(audio_data_list)
        # Ensure the concatenated audio has the correct shape
        if concatenated_audio.ndim == 1:
            concatenated_audio = concatenated_audio.reshape(-1, 1)
        return concatenated_audio, sample_rate
    else:
        return None, None

def generate_audio(sentences, query_uuid, audio_queue, concatenation_queue):
    for sentence in sentences:
        tts_result = tts_client.predict(
            sentence,
            os.getenv("TTS_VOICE"),
            float(os.getenv("TTS_SPEED")),
            os.getenv("TTS_LANG"),
            api_name="/gen_voice"
        )
        
        with open(tts_result, "rb") as f:
            audio_data = f.read()
        data, sample_rate = librosa.load(io.BytesIO(audio_data), sr=None)
        speed_factor = float(os.getenv("TTS_SPEED", 0.5))
        stretched_data = librosa.effects.time_stretch(data, rate=1/speed_factor)
        audio_queue.put((stretched_data, sample_rate))
        concatenation_queue.put((stretched_data, sample_rate))
    
    audio_queue.put(None)
    concatenation_queue.put(None)

def playback_audio(audio_queue, speak_callback):
    while True:
        audio_data = audio_queue.get()
        if audio_data is None:
            break
        stretched_data, sample_rate = audio_data
        speak_callback(stretched_data, sample_rate)

def process_sentence(sentence, query_uuid, audio_queue):
    tts_result = tts_client.predict(
        sentence,
        os.getenv("TTS_VOICE"),
        float(os.getenv("TTS_SPEED")),
        os.getenv("TTS_LANG"),
        api_name="/gen_voice"
    )
    
    with open(tts_result, "rb") as f:
        audio_data = f.read()
    data, sample_rate = librosa.load(io.BytesIO(audio_data))
    speed_factor = float(os.getenv("TTS_SPEED", 0.5))
    stretched_data = librosa.effects.time_stretch(data, rate=1/speed_factor)
    audio_queue.put((stretched_data, sample_rate))

def playback_audio(audio_queue, speak_callback):
    while True:
        audio_data = audio_queue.get()
        if audio_data is None:
            break
        stretched_data, sample_rate = audio_data
        speak_callback(stretched_data, sample_rate)```
---

./main.py
```
import pygame
import threading
from modules.conversate_app import ConversateApp
from modules.database import create_interactions_table, check_missing_journal_entries
from modules.api import app  # Import the app variable from modules.api
from modules.llm_api import process_query

def main():
    create_interactions_table()
    check_missing_journal_entries(process_query)
    # Initialize Pygame
    pygame.init()

    # Set up the display
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Conversate")

    # Create an instance of ConversateApp
    conversate_app = ConversateApp(screen)  # Rename the variable to avoid confusion

    print("Ready.")
    # Run the application
    conversate_app.run()

    # Clean up
    pygame.quit()

def run_api():
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    api_thread = threading.Thread(target=run_api)
    api_thread.start()
    main()```
---

./modules/api.py
```

from flask import Flask, request, jsonify
from flask_httpauth import HTTPBasicAuth
from .llm_api import process_query
from .database import log_interaction, get_last_messages
import os
import re
import json
import uuid
from datetime import datetime


app = Flask(__name__)
auth = HTTPBasicAuth()

API_PASSWORD = 'lookatmeimanapikey'

@auth.verify_password
def verify_password(username, password):
    return password == API_PASSWORD

@app.route('/api/query', methods=['POST'])
@auth.login_required
def query_api():
    data = request.get_json()
    query_text = data.get('query', '')

    if not query_text:
        return jsonify({'error': 'Query text is missing'}), 400

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

    return jsonify({'short_answer': short_answer})```
---
