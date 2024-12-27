from flask import Flask
from flask_restful import Api

api = Api()

def create_app():
    app = Flask(__name__)

    from app.path import AI_API_PATH

    AI_API_PATH()

    api.init_app(app)

    return app