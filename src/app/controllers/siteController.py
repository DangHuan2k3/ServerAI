import os
from flask import Flask, Response, send_file
import json
from util.AI.main import prediction

from pydub import AudioSegment


class SiteController:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def getLabel(request):
        print(request.files)
        audio_file = request.files['audio']
        print(audio_file)
        if audio_file.filename == '':
            data = {"text": "NOT FOUND FILES AUDIO/WAV"}
            return Response(response=json.dumps(data), status=304,
                            mimetype='application/json')
        print(os.getcwd())
        audio_file.save(
            os.getcwd() + ('\\src\\resources\\audio\\output.m4a'))
        # Send .m4a file and convert to wav
        sound = AudioSegment.from_file(
            os.getcwd() + ('\\src\\resources\\audio\\output.m4a'), format='m4a')
        sound.export(
            os.getcwd() + ('\\src\\resources\\audio\\output.wav'), format='wav')
        label = prediction()
        data = {"label": label}
        print(data)
        resp = Response(response=json.dumps(data), status=200,
                        mimetype='application/json')
        return resp
