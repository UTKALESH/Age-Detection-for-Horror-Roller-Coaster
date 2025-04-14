import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import datetime
import csv
import os
import time
import math

MODEL_PATH = 'saved_models_scratch/age_gender_model.keras'
LOG_FILE = 'detected_entries.csv'

IMG_WIDTH = 64
IMG_HEIGHT = 64
CHANNELS = 3

AGE_MIN_ALLOWED = 13
AGE_MAX_ALLOWED = 60
AGE_BIN_LABELS = ['0-12', '13-19', '20-29', '30-39', '40-49', '50-60', '61+']
GENDER_LABELS = ['Male', 'Female'] 

COLOR_ALLOWED = (0, 255, 0)
COLOR_NOT_ALLOWED = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2

def preprocess_face(face_img, target_width, target_height, channels):
    try:
        img_resized = cv2.resize(face_img, (target_width, target_height), interpolation=cv2.INTER_AREA)
        if channels == 1:
            if len(img_resized.shape) == 3:
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            else: img_gray = img_resized
            img_processed = np.expand_dims(img_gray, axis=-1)
        else:
            if len(img_resized.shape) == 2: img_processed = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
            elif img_resized.shape[2] == 1: img_processed = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
            else: img_processed = img_resized

        img_normalized = img_processed.astype('float32') / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        return img_batch
    except Exception as e:
        print(f"Error preprocessing face: {e}")
        return None

def setup_csv(filename):
    file_exists = os.path.isfile(filename)
    try:
        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists or os.path.getsize(filename) == 0:
                header = ['Timestamp', 'Predicted Age Range', 'Predicted Gender', 'Status']
                writer.writerow(header)
                print(f"Initialized log file: {filename}")
    except IOError as e:
        print(f"Error setting up CSV file {filename}: {e}")
        return False
    return True

def log_data(filename, age_range, gender, status):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([timestamp, age_range, gender, status])
    except IOError as e:
        print(f"Error writing to CSV file {filename}: {e}")

print("--- Roller Coaster Age/Gender Detector ---")
print("--- Using MediaPipe for FACE DETECTION ONLY ---")
print("--- Using Custom Trained Model for Age/Gender Prediction ---")

try:
    print("Loading custom trained age/gender model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load custom model from {MODEL_PATH}")
    print(f"Ensure training was completed and the model file exists. Error: {e}")
    exit()

print("Initializing MediaPipe Face Detection...")
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

print("Initializing video capture...")
cap = cv2.VideoCapture(0)
time.sleep(1.0)

if not cap.isOpened():
    print("FATAL ERROR: Could not open video capture device.")
    exit()

print("Starting real-time detection... Press 'q' to quit.")

if not setup_csv(LOG_FILE):
     print("Warning: Proceeding without CSV logging due to setup error.")

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame. Exiting.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)
    frame_height, frame_width, _ = frame.shape

    if results.detections:
        for detection in results.detections:
            try:
                bboxC = detection.location_data.relative_bounding_box
                xmin = int(bboxC.xmin * frame_width)
                ymin = int(bboxC.ymin * frame_height)
                width = int(bboxC.width * frame_width)
                height = int(bboxC.height * frame_height)

                x1, y1 = max(0, xmin), max(0, ymin)
                x2, y2 = min(frame_width - 1, xmin + width), min(frame_height - 1, ymin + height)

                if x1 >= x2 or y1 >= y2: continue 

                face_roi = frame[y1:y2, x1:x2]

                face_batch = preprocess_face(face_roi, IMG_WIDTH, IMG_HEIGHT, CHANNELS)
                if face_batch is None: continue

                predictions = model.predict(face_batch, verbose=0)
                age_preds = predictions[0]
                gender_preds = predictions[1]

                pred_age_bin_index = np.argmax(age_preds[0])
                pred_gender_prob = gender_preds[0][0]
                pred_age_range = AGE_BIN_LABELS[pred_age_bin_index]
                pred_gender_index = 1 if pred_gender_prob > 0.5 else 0
                pred_gender = GENDER_LABELS[pred_gender_index]

                confidence_age = np.max(age_preds[0]) * 100
                confidence_gender = (pred_gender_prob if pred_gender_index==1 else 1-pred_gender_prob) * 100

                is_allowed = True
                status_message = "Allowed"
                box_color = COLOR_ALLOWED

                if pred_age_bin_index == 0 or pred_age_bin_index == (len(AGE_BIN_LABELS) - 1):
                    is_allowed = False
                    status_message = "Not Allowed"
                    box_color = COLOR_NOT_ALLOWED

                log_data(LOG_FILE, pred_age_range, pred_gender, status_message)

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                label_age = f"Age: {pred_age_range}"
                label_gender = f"Gender: {pred_gender}"
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + height + 20
                cv2.putText(frame, label_age, (x1, text_y), FONT, FONT_SCALE, box_color, FONT_THICKNESS)
                cv2.putText(frame, label_gender, (x1, text_y + 20), FONT, FONT_SCALE, box_color, FONT_THICKNESS)
                if not is_allowed:
                    cv2.putText(frame, status_message, (x1, text_y + 40), FONT, FONT_SCALE*1.1, box_color, FONT_THICKNESS+1)

            except Exception as pred_err:
                print(f"Error processing detection: {pred_err}")

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), FONT, 0.7, (255, 255, 255), FONT_THICKNESS)

    cv2.imshow('Roller Coaster Security Cam - Press Q to Quit', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        print("Exit key pressed.")
        break

print("Releasing resources...")
cap.release()
cv2.destroyAllWindows()
face_detection.close()
print("Application finished.")