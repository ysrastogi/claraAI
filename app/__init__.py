from flask import Flask
from .main import main
from .extensions import llm 

def create_app():
    app = Flask(__name__)
    app.register_blueprint(main)
    return app