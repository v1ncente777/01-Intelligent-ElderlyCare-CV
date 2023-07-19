
import json
from flask import Flask, request


app = Flask(__name__)


@app.route('/back/test', methods=["POST"])
def collecting_faces():
    data = json.loads(request.get_data(as_text=True))
    print(data)
    return {'data': 'success'}


if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=9000, host="127.0.0.1")
