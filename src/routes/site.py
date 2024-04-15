from markupsafe import escape
from app.controllers.siteController import SiteController


def routing(path, request):
    params = path.split('/')
    # print(params)
    print(request.method)

    if request.method == 'POST':
        # [POST] /audio
        if params[0] == 'audio':
            return SiteController.getLabel(request)
    elif request.method == 'GET':
        return 'GET'

    return f'404 NOT FOUND {escape(path)}'

    # return f"Hello world {escape(path)}"
