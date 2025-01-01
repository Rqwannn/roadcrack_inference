from flask import Flask
from flask_restful import Api
from flask_cors import CORS
from flask_talisman import Talisman

api = Api()

def create_app():
    app = Flask(__name__)

    from app.path import AI_API_PATH

    AI_API_PATH()

    Talisman(app, content_security_policy=None, force_https=True, strict_transport_security=True)
    CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)

    api.init_app(app)

    return app