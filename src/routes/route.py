from routes import site
from flask import request


def init(app):
    @app.route("/", methods=['POST', 'GET'])
    def indexRouting():
        return site.routing('/', request)

    @app.route("/<path:subpath>", methods=['POST', 'GET'])
    def siteRouting(subpath):
        return site.routing(subpath, request)
