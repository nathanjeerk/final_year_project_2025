import cv2
import threading
import mediapipe as mp
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load('/mnt/data/LogisticRegression.pkl')  # Update the path to your .pkl file

latest_frame = None
latest_label = "LIVE FEED ONLY"

def get_latest_frame():
    return latest_frame

def get_latest_label():
    return latest_label

def start_pose_detection():
    def run():
        global latest_frame, latest_label

        # Use MediaPipe just for drawing (optional)
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic

        cap = cv2.VideoCapture(0)

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.resize(frame, (640, 480))

                # Optional: draw landmarks
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = holistic.process(image_rgb)
                
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                    # Extract pose landmarks as features
                    landmarks = results.pose_landmarks.landmark
                    features = np.array([landmark.x for landmark in landmarks] + 
                                        [landmark.y for landmark in landmarks] + 
                                        [landmark.z for landmark in landmarks])

                    # Reshape features to match model input shape
                    features = features.reshape(1, -1)

                    # Predict using the loaded model
                    prediction = model.predict(features)
                    latest_label = prediction[0]  # Assign the prediction to the latest label

                latest_frame = frame.copy()

    threading.Thread(target=run, daemon=True).start()
