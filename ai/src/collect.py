# scripts/collect.py
import os
import cv2
import csv
import yaml
import numpy as np
import mediapipe as mp

# Load config
with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

VIDEO_BASE_DIR = config["data"]["video_base_dir"]
OUTPUT_CSV = config["data"]["output_csv"]
CLASSES = config["data"]["classes"]
RESOLUTION = tuple(config["video"]["resolution"])
DETECTION_CONF = config["mediapipe"]["detection_confidence"]
TRACKING_CONF = config["mediapipe"]["tracking_confidence"]

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic




def process_video(video_path, label):
    """Process a video and extract pose landmarks"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return

    with mp_holistic.Holistic(
        min_detection_confidence=DETECTION_CONF,
        min_tracking_confidence=TRACKING_CONF
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, RESOLUTION)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            results = holistic.process(img)
            img.flags.writeable = True

            if results.pose_landmarks:
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose]).flatten())
                row = [label] + pose_row
                with open(OUTPUT_CSV, "a", newline="") as f:
                    csv.writer(f).writerow(row)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            cv2.imshow(f"Processing: {label}", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def setup_csv():
    """Create CSV file with headers if not exists"""
    if not os.path.exists(OUTPUT_CSV):
        header = ["label"]
        for i in range(33):  # 33 pose landmarks
            for coord in ["x", "y", "z", "visibility"]:
                header.append(f"pose_{i}_{coord}")
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        with open(OUTPUT_CSV, "w", newline="") as f:
            csv.writer(f).writerow(header)    


def process_all():
    setup_csv()
    for label in CLASSES:
        folder_path = os.path.join(VIDEO_BASE_DIR, label)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist")
            continue

        print(f"Processing label: {label}")
        for file in os.listdir(folder_path):
            if file.endswith(('.mp4', '.avi', '.mov')):
                print(f"  - {file}")
                process_video(os.path.join(folder_path, file), label)
    print("Finished processing all videos.")


if __name__ == "__main__":
    process_all()
