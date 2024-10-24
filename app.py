import cv2
from flask import Flask, Response, request, jsonify
from embeddings_db import save_embedding_to_db, load_embeddings
from video_recognition import CameraStream, ProcessFrameStream
from deepface import DeepFace
import base64
import numpy as np

app = Flask(__name__)

# متغير لتخزين حالة الكاميرا
camera = None
url = 'http://192.168.8.159:8080/video'

stream = CameraStream(src=url)
process_stream = ProcessFrameStream(stream, load_embeddings())

def open_camera():
    stream.start()
    process_stream.start()

def release_camera():
    process_stream.stop()
    stream.stop()

def generate_frames():
    open_camera()  # فتح الكاميرا إذا كانت مغلقة
    while True:
        frame = process_stream.render()
        if frame is not None:
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

            # تحويل الإطار إلى تنسيق JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()  # تحويل البيانات إلى بايت

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # بث الإطار

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed')
def stop_feed():
    release_camera()  # تحرير الكاميرا عند الخروج من صفحة البث
    return "Camera released and streaming stopped."

def base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.route('/api/extract-features', methods=['POST'])
def extract_features():
    data = request.get_json()
    embeddings = []
    try:
        for image in data['images']:
            try:
                face_img_rgb = cv2.cvtColor(base64_to_image(image), cv2.COLOR_BGR2RGB)
                embeddings.append(DeepFace.represent(face_img_rgb, model_name='Facenet')[0]["embedding"])
            except Exception as e:
                print(f'error: {str(e)}')
        if len(embeddings) <= 0:
            raise("no embeddings")
        save_embedding_to_db(data['userId'], {"name": data['name']}, embeddings)
        return jsonify({'embeddings': embeddings, 'name': data['name'], 'userId': data['userId']})
    except Exception as e:
        return jsonify({'error': str(e)})
