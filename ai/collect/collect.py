import mediapipe as mp
import cv2, csv, os
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Config
VIDEO_BASE_DIR = "./videos" # base directory contain exercises folder
EXERCISES = ["pullups", "jump", "squat"] # list of exercise folders
OUTPUT_CSV = "coordinate.csv" # output csv file

def process_video(video_path, exercise_name):
    # process only single video and save landmarks to csv
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False

            results = holistic.process(img)

            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            try:
                if results.pose_landmarks:
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    # adding exercises name as the first column
                    row = [exercise_name] + pose_row

                    with open(OUTPUT_CSV, mode = "a", newline = "") as f:
                        csv_writer = csv.writer(f, delimiter = ",", quotechar = '"', quoting = csv.QUOTE_MINIMAL)
                        csv_writer.writerow(row)
            except Exception as e:
                print(f"Error processing frame: {e}")

            cv2.imshow(f"Processing: {exercise_name}", img)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
    capture.release()
    cv2.destroyAllWindows()

def setup_csv():
    # prepare csv file with headers
    if not os.path.exists(OUTPUT_CSV):
        header = ["exercise_name"]
        for landmark_num in range(33):
            for coordinate in ["x", "y", "z", "visibility"]:
                header.append(f"pose_{landmark_num}_{coordinate}")
        
        with open(OUTPUT_CSV, mode = "w", newline = "") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(header)

def process_all_videos():
    setup_csv()

    for exercise in EXERCISES:
        exercise_dir = os.path.join(VIDEO_BASE_DIR, exercise)

        if not os.path.exists(exercise_dir):
            print(f"Warning: Exercise directory {exercise_dir} not found")
            continue
    
        print(f"Processing {exercise} videos...")
        video_files = [f for f in os.listdir(exercise_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

        for video_file in video_files:
            video_path = os.path.join(exercise_dir, video_file)
            print(f"Processing {video_file}...")
            process_video(video_path, exercise)
    print("All videos processed!")

if __name__ == "__main__":
    process_all_videos()