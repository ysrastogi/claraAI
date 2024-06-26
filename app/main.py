from flask import Flask,Blueprint, request, jsonify, send_from_directory
from langchain_core.prompts.prompt import PromptTemplate
import logging
import uuid
import threading
import json
import os
from .extensions import llm,sessions 
from langchain.chains import LLMChain
from werkzeug.utils import secure_filename
from app.utils.pdf_extractor import process_text_file_and_store_vectors

main = Blueprint('main', __name__)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_ai_request(session_id, message):
    # AI processing
    # Substitute `model_prompter(message)` with your desired LLM's completion method
    prompt = PromptTemplate.from_template(message)
    chain = LLMChain(llm = llm, prompt = prompt)
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

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Process the file
        file_paths = [file_path]
        percentile = 96  # Example percentile
        vectorstore_path = 'vectorstore.faiss'  # Example path for saving the vectorstore
        process_text_file_and_store_vectors(file_paths, percentile, vectorstore_path)
        return jsonify({'message': 'File successfully uploaded and processed'}), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400