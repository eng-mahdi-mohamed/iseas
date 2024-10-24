# Face Recognition & Emotion Analysis Project

This project uses DeepFace, Mediapipe, and FER to perform real-time face recognition and emotion analysis.

## Setup Instructions

1. **Clone the repository**:
    ```
    git clone https://your-repo-url
    cd face_recognition_project
    ```

2. **Create a virtual environment**:
    - For Linux/Mac:
      ```
      python3 -m venv venv
      source venv/bin/activate
      ```
    - For Windows:
      ```
      python -m venv venv
      venv\Scripts\activate
      ```

3. **Install the required libraries**:
    ```
    pip install -r requirements.txt
    ```

4. **Place student images**:
   Add student images to the `data/students/` folder. Each image file should be named with the student ID.

5. **Run the project**:
    ```
    python video_recognition.py
    ```

## Notes
- Make sure to adjust the threshold for face recognition and emotion analysis as needed.
