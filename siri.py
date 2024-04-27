from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain.prompts import FixedPrompt
from langchain.schema import TextCompletionRequest
from flask import Flask, request, jsonify
import logging
import uuid
import threading
import json
# In addition import all you may need for your API call

app = Flask(__name__)

# In-memory storage for simplicity
sessions = {}
llm = Ollama(model="llama3")
# Create a fixed prompt for the model
prompt = FixedPrompt(prompt_text="Translate the following text to French: ")

# Create a chain using the Ollama model and the fixed prompt
chain = LLMChain(llm=llm, prompt=prompt)

def process_ai_request(session_id, message):
    # AI processing
    # Substitute `model_prompter(message)` with your desired LLM's completion method
    response = model_prompter(message)

    # Update the session with the response
    sessions[session_id]["status"] = "complete"
    sessions[session_id]["response"] = response

@app.route('/siri', methods=['POST'])
def siri_endpoint():
    logging.info("Siri endpoint was called")
    data = request.json
    message = data['message']

    # Generate a unique session ID
    session_id = str(uuid.uuid4())

    # Initialize session status
    sessions[session_id] = {"status": "processing", "response": None}

    # Start AI processing in a separate thread
    ai_thread = threading.Thread(target=process_ai_request, args=(session_id, message))
    ai_thread.start()

    # Return the session ID immediately
    return jsonify({"session_id": session_id})

def log_response(response_data):
    # Log the response data in a pretty-printed JSON format
    app.logger.info(json.dumps(response_data, indent=4))

@app.route('/siri', methods=['GET'])
def check_status():
    session_id = request.args.get('session_id')
    session = sessions.get(session_id, None)

    if not session:
        response_data = {"error": "Invalid session ID"}
        log_response(response_data)  # Log the error response
        return jsonify(response_data), 404

    response_data = {"status": session["status"], "response": session.get("response")}
    log_response(response_data)  # Log the successful response
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)