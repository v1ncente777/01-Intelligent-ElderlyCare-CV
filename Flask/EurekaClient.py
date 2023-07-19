from threading import Thread

import keras
import cv2
import py_eureka_client.eureka_client as eureka_client
import json
from flask import Flask, request, Response
from flask_cors import CORS

from checkng_fence import CheckFence
from trainingfacerecognition import train_face
from checking_fall import CheckFall
from collecting_faces import CollectingFacesThread
# from work16 import StrangersExpressionThread
from stranger_expression import StrangersExpression
from checkng_fence import CheckFence
from checking_activity import CheckActivity
from util import http_post, message_queue, oss_download

# global variable
cv_state = 'Available'  # Avalilable Using Updating
video_camera = None
global_frame = None
application_name = 'wrx'
server_ip = '192.168.31.55'
global_server_host = "192.168.31.162"
global_server_port = 5000
eureka_server_url = 'http://' + server_ip + ':8761/eureka'
gateway_url = 'http://' + server_ip + ':8080/springboot'
absolute_path = 'C:\\Users\\yukino\\Desktop\\elderly-care-backend-master\\elderly-care-backend-master\\Flask\\images\\oldpeople\\101'
location = 'room'
if location not in ['room', 'yard', 'corridor', 'desk']:
    raise ValueError('location must be one of room, yard, corridor or desk')


# register to eureka server
def set_eureka():
    server_host = global_server_host
    server_port = global_server_port
    eureka_client.init(eureka_server=eureka_server_url,
                       app_name=application_name,
                       # 当前组件的主机名，可选参数，如果不填写会自动计算一个，如果服务和 eureka 服务器部署在同一台机器，请必须填写，否则会计算出 127.0.0.1
                       instance_host=server_host,
                       instance_port=server_port,
                       # 调用其他服务时的高可用策略，可选，默认为随机
                       ha_strategy=eureka_client.HA_STRATEGY_RANDOM)


def video_stream():
    global video_camera
    global global_frame

    while True:
        if video_camera is None:
            return
        frame = video_camera.get_frame()

        if frame is not None:
            global_frame = frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')


app = Flask(__name__)
CORS(app, supports_credentials=True)

set_eureka()


@app.route('/')
def index():
    return {'data': 'welcome to flask'}


@app.route('/info')
def info():
    return {'data': 'welcome to flask'}


@app.route('/status', methods=['GET'])
def status():
    return {"status": cv_state}


@app.route('/update', methods=['GET'])
def update():
    oss_download('models/model_version.txt', 'temp/model_version.txt')
    f = open('temp/model_version.txt', 'r')
    content = f.read()
    f.close()
    local_version = int(content[11:12])
    print(local_version)
    f = open('models/model_version.txt', 'r')
    content = f.read()
    f.close()
    remote_version = int(content[11:12])
    print(remote_version)
    latest_version = -1
    if remote_version >= local_version:
        latest_version = remote_version
    else:
        latest_version = local_version
    if remote_version > local_version:
        oss_download('models/face_recognition_hog.pickle', 'models/face_recognition_hog.pickle')
        oss_download('models/model_version.txt', 'models/model_version.txt')
    else:
        latest_version += 1
        f = open('models/model_version.txt', 'w')
        f.write('is_version=' + str(latest_version))
        f.close()
        thread = Thread(target=train_face)
        thread.start()
    # train_face()
    return 'Update OK'


@app.route('/close', methods=['GET'])
def close():
    global cv_state
    global video_camera
    del video_camera
    video_camera = None
    cv2.destroyAllWindows()
    cv_state = 'Available'
    return {"status": cv_state}


@app.route('/message', methods=['GET'])
def message():
    temp = ''
    try:
        temp = message_queue.get(block=False)
    except:
        return "nothing"
    finally:
        print(temp)
        return temp


@app.route('/collecting_faces', methods=["POST"])
def collecting_faces():
    cv2.destroyAllWindows()
    data = json.loads(request.get_data(as_text=True))
    print(data)
    uid = ''
    my_type = ''
    for key, value in data.items():
        if key == 'id':
            uid = value
        if key == 'type':
            my_type = value
    collecting_faces_thread = CollectingFacesThread(absolute_path, my_type, uid)
    collecting_faces_thread.start()
    return {'data': 'success'}


@app.route('/stranger_expression', methods=["GET", "POST"])
def stranger_expression():
    global cv_state
    if cv_state is not 'Available':
        return {"data": "camera is not available"}
    keras.backend.clear_session()
    global video_camera
    cv_state = 'using'
    url = 'http://' + server_ip + ':9000/back/eventLoad'
    video_camera = StrangersExpression(url)
    return Response(video_stream(), status=200,
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/checking_fall', methods=["GET", "POST"])
def checking_fall():
    global cv_state
    if cv_state is not 'Available':
        return {"data": "camera is not available"}
    keras.backend.clear_session()
    global video_camera
    cv_state = 'using'
    url = 'http://' + server_ip + ':9000/back/eventLoad'
    video_camera = CheckFall(url)
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/checking_fence', methods=["GET", "POST"])
def checking_fence():
    global cv_state
    if cv_state is not 'Available':
        return {"data": "camera is not available"}
    keras.backend.clear_session()
    global video_camera
    cv_state = 'using'
    url = 'http://' + server_ip + ':9000/back/eventLoad'
    video_camera = CheckFence(url)
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/checking_activity', methods=["GET", "POST"])
def checking_activity():
    global cv_state
    if cv_state is not 'Available':
        return {"data": "camera is not available"}
    keras.backend.clear_session()
    global video_camera
    cv_state = 'using'
    url = 'http://' + server_ip + ':9000/back/eventLoad'
    video_camera = CheckActivity(url)
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/test')
def test():
    url = 'http://' + server_ip + ':9000/back/test'
    res = http_post(url, event_type=2, event_location='room', event_desc='asd')
    print(res)
    res = http_post(url, event_type=2, event_location='room', event_desc='asd', old_people_id='0001')
    print(res)
    return {'data': 'welcome to flask'}


if __name__ == '__main__':
    app.run(debug=False, threaded=True, port=5000, host="0.0.0.0")
