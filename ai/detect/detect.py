import cv2
import mediapipe as mp
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open("best_workout_pose_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

capture = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        # Process image with MediaPipe
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = holistic.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        try:
            if results.pose_landmarks:
                # Extract pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array(
                    [[landmark.x, landmark.y, landmark.z, landmark.visibility] 
                     for landmark in pose]
                ).flatten())
                
                # Convert to DataFrame with correct shape (1 row, n features)
                X = pd.DataFrame([pose_row])
                
                # Make prediction
                workout_pose_class = le.inverse_transform(model.predict(X))[0]
                workout_pose_prob = model.predict_proba(X)[0]
                
                # Get coordinates for text placement (using nose instead of ear)
                nose_coords = tuple(np.multiply(
                    np.array((
                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x,
                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y
                    )), [frame.shape[1], frame.shape[0]]
                ).astype(int))

                # Display class and probability
                cv2.rectangle(img, 
                             (nose_coords[0], nose_coords[1] + 5), 
                             (nose_coords[0] + len(workout_pose_class) * 20, nose_coords[1] - 30), 
                             (245, 117, 16), -1)
                cv2.putText(img, workout_pose_class, 
                           (nose_coords[0], nose_coords[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Status box
                cv2.rectangle(img, (0, 0), (250, 60), (245, 117, 16), -1)
                
                # Display class
                cv2.putText(img, 'CLASS', (95, 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img, workout_pose_class, (90, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display probability
                cv2.putText(img, 'PROB', (15, 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img, str(round(max(workout_pose_prob), 2)), (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")  # Print error for debugging

        cv2.imshow("AI Workout Analyzer", img)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

capture.release()
cv2.destroyAllWindows()