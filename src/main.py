from flask import Flask
from markupsafe import escape
from routes import route

app = Flask(__name__)

route.init(app)
