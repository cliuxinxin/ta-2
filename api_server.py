from gevent import monkey
monkey.patch_all()
from flask import Flask, request
from gevent import pywsgi
import re
from time import time
import json
import sync
from flask_cors import CORS

app = Flask(__name__)
app.debug = False 
CORS(app, supports_credentials=True)

class BadRequest(Exception):
    """将本地错误包装成一个异常实例供抛出"""
    def __init__(self, message, status=400, payload=None):
        self.message = message
        self.status = status
        self.payload = payload

@app.errorhandler(BadRequest)
def handle_bad_request(error):
    """捕获 BadRequest 全局异常，序列化为 JSON 并返回 HTTP 400"""
    payload = dict(error.payload or ())
    payload['status'] = error.status
    payload['message'] = error.message
    return  json.dumps(payload), 600

@app.route('/', methods=['get'])
def index():
    start = time()
    dict={}
    sentences = request.args.get('sentences')
    sentences = sentences + '。'
    fields = re.split(r'(。|，|)', sentences)
    sentences = list(filter(None,fields[::2]))
    delimiters = fields[1::2]
    try:
        output_file = sync.synic(sentences,delimiters)
    except Exception as e:
        raise BadRequest(str(e), 600)
    dict['wav_file'] = output_file.replace('.wav', '')
    stop = time()
    dict['time'] = stop - start
    dict['status'] = 200
    return json.dumps(dict),200


if __name__ == "__main__":
    server = pywsgi.WSGIServer(('0.0.0.0', 19877), app)
    print("start to serve")
    server.serve_forever()
