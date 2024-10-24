import cv2
import mediapipe as mp
from deepface import DeepFace
from fer import FER
import threading
import numpy as np
from datetime import datetime

# Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# إعداد مرشح كالمان
kalman_filter = cv2.KalmanFilter(4, 2)
kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman_filter.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

def apply_kalman_filter(x, y, w, h):
    measurement = np.array([[np.float32(x + w / 2)], [np.float32(y + h / 2)]])
    kalman_filter.correct(measurement)
    predicted = kalman_filter.predict()
    predicted_x = predicted[0] - w / 2
    predicted_y = predicted[1] - h / 2
    return int(predicted_x), int(predicted_y), w, h


# FER emotion detection
emotion_detector = FER()

class CameraStream:
    def __init__(self, src=0):
        self.src = src
        self.frame_changed = threading.Event()
        self.cap = None
        self.frame = None
        self.stopped = False
        self.running = False

    def update(self):
        self.running = True
        try:
            while not self.stopped and self.cap is not None and self.cap.isOpened():
                grabbed, frame = self.cap.read()
                if grabbed:
                    self.frame = frame
                    self.frame_changed.set()
        except:
            print('error: camera read')
        self.running = False

    def read(self):
        return self.frame
    
    def start(self):
        self.stopped = False
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.src)
        if not self.running:
            threading.Thread(target=self.update, args=()).start()

    def stop(self):
        self.stopped = True
        self.cap.release()
        self.running = False


class ProcessFrameStream:
    def __init__(self, stream, known_faces):
        self.stream = stream
        self.known_faces = known_faces
        self.student_face_data = {}
        self.stopped = False
        self.running = False

    def start(self):
        self.stopped = False
        if not self.running:
            threading.Thread(target=self.process, args=()).start()

    def stop(self):
        self.student_face_data = {}
        self.running = False
        self.stopped = True

    def process(self):
        self.running = True
        while not self.stopped:
            self.stream.frame_changed.wait()
            self.stream.frame_changed.clear()
            frame = self.stream.read()
            if frame is not None:
                self.process_frame(frame)
        self.running = False

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        current_face_data = {}

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x, y, w, h = self.get_bounding_box(bbox, frame)

                if w > 0 and h > 0:
                    face_img = frame[y:y + h, x:x + w]
                    face_id = self.recognize_face_by_bounds((x, y, w, h))
                    distance = None
                    if face_id is None:
                        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        face_id, distance = self.recognize_face(face_img_rgb)

                    if face_id:
                        emotion_label, confidence = emotion_detector.top_emotion(face_img)
                        if emotion_label is None and face_id in self.student_face_data:
                            (emotion_label, confidence) = self.student_face_data[face_id]["emotion"]
                        if distance is None and face_id in self.student_face_data:
                            distance = self.student_face_data[face_id]["distance"]
                        current_face_data[face_id] = {"emotion": (emotion_label, confidence), "bounds": (x, y, w, h), "distance": distance}

        self.student_face_data = current_face_data

    def recognize_face(self, face_img_rgb):
        face_embedding = []
        try:
            face_embedding = DeepFace.represent(img_path=face_img_rgb, model_name="Facenet")[0]["embedding"]
        except Exception as e:
            print(f"Error recognizing face: {e}")
            return (None, None)
        
        min_distance = float('inf')
        best_match = None
        for student_id, face in self.known_faces.items():
            avg_embedding = np.mean(face['embeddings'], axis=0)

            distance = np.linalg.norm(np.array(face_embedding) - np.array(avg_embedding))
                
            if distance < min_distance:
                min_distance = distance
                best_match = student_id
            
        if min_distance < 0.6:
            return best_match
        
        return (best_match, min_distance)

    def recognize_face_by_bounds(self, bounds):
        x, y, width, height = bounds
        center_x = x + width / 2
        center_y = y + height / 2

        for face_id, data in self.student_face_data.items():
            last_x, last_y, last_width, last_height = data["bounds"]
            last_center_x = last_x + last_width / 2
            last_center_y = last_y + last_height / 2

            distance = np.sqrt((center_x - last_center_x) ** 2 + (center_y - last_center_y) ** 2)
            if distance < 50:
                return face_id

        return None

    def get_bounding_box(self, bbox, frame):
        h, w, _ = frame.shape
        scaleFactor = 1.8
        width = bbox.width * w * scaleFactor
        height = bbox.height * h * scaleFactor
        x = (bbox.xmin * w) - (width / 2)
        y = (bbox.ymin * h) - (height / 2)
        
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        # x, y, width, height = apply_kalman_filter(x, y, width, height)
        return max(0, x), max(0, y), min(w - x, width), min(h - y, height)

    def render(self):
        frame = self.stream.read()
        if frame is not None:
            for student_id, data in self.student_face_data.items():
                x, y, w, h = data["bounds"]
                emotion_label, confidence = data["emotion"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'{self.known_faces[f"{student_id}"]["user"]["name"]}:{round(data["distance"], 2)}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                if emotion_label is not None:
                    cv2.putText(frame, f"{emotion_label} ({confidence:.2f})", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                
            return frame