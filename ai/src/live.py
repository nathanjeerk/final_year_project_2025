import cv2
import numpy as np
import yaml
import joblib
import mediapipe as mp
import os

# === Load config ===
with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = os.path.join(config['model']['save_dir'], config['model']['save_filenames']['LogisticRegression'])
LABEL_ENCODER_PATH = os.path.join(config['model']['save_dir'], "label_encoder.pkl")
RESOLUTION = tuple(config["video"]["resolution"])
DETECTION_CONF = config["mediapipe"]["detection_confidence"]
TRACKING_CONF = config["mediapipe"]["tracking_confidence"]

# === Load model and LabelEncoder ===
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
LABELS = list(label_encoder.classes_)

# === Setup MediaPipe ===
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=DETECTION_CONF,
                          min_tracking_confidence=TRACKING_CONF) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.resize(frame, RESOLUTION)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True

        # Draw landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Extract pose
        if results.pose_landmarks:
            pose = results.pose_landmarks.landmark
            pose_row = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose]).flatten()

            # Check shape
            if pose_row.shape[0] != 132:
                print(f"Incomplete pose detected (got {pose_row.shape[0]} features), skipping.")
                continue

            # Predict
            prediction = model.predict(pose_row.reshape(1, -1))[0]
            label = label_encoder.inverse_transform([prediction])[0]

            print(f"Prediction: {prediction}, Label: {label}")

            # Display label
            color = (0, 255, 0) if "good" in label else (0, 0, 255)
            cv2.putText(frame, label.upper(), (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        else:
            cv2.putText(frame, "NO POSE DETECTED", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        cv2.imshow('Squat Detector', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
