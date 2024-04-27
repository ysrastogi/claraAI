from flask import Blueprint, request, jsonify
from langchain_core.prompts.prompt import PromptTemplate
import logging
import uuid
import threading
import json
from .extensions import llm, chain, sessions  # Assuming these are initialized in a separate module
from langchain.chains import LLMChain

main = Blueprint('main', __name__)

def process_ai_request(session_id, message):
    # AI processing
    # Substitute `model_prompter(message)` with your desired LLM's completion method
    prompt = PromptTemplate.from_template(message)
    chain = LLMChain(llm = llm, prompt = prompt)\
    response = chain.invoke()
    # Update the session with the response
    sessions[session_id]["status"] = "complete"
    sessions[session_id]["response"] = response

@main.route('/siri', methods=['POST'])
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
    logging.info(json.dumps(response_data, indent=4))

@main.route('/siri', methods=['GET'])
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