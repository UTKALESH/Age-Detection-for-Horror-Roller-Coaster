import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import datetime
import csv
import os
import time

MODEL_PATH = 'saved_model/age_gender_model.keras'
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
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
FONT_SCALE = 0.7
FONT_THICKNESS = 2

def preprocess_face(face_img, target_width, target_height, channels):
    try:
        img_resized = cv2.resize(face_img, (target_width, target_height))

        if channels == 1:
            if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img_resized
            img_processed = np.expand_dims(img_gray, axis=-1)
        else:
            if len(img_resized.shape) == 2:
                 img_processed = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
            elif img_resized.shape[2] == 1:
                 img_processed = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
            else:
                 img_processed = img_resized

        img_normalized = img_processed.astype('float32') / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        return img_batch
    except Exception as e:
        print(f"Error preprocessing face: {e}")
        return None


def setup_csv(filename):
    file_exists = os.path.isfile(filename)
    try:
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists or os.path.getsize(filename) == 0:
                header = ['Timestamp', 'Predicted Age Range', 'Predicted Gender', 'Status']
                writer.writerow(header)
                print(f"Created or found empty log file: {filename}")
    except IOError as e:
        print(f"Error setting up CSV file {filename}: {e}")
        return False
    return True

def log_data(filename, age_range, gender, status):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([timestamp, age_range, gender, status])
    except IOError as e:
        print(f"Error writing to CSV file {filename}: {e}")


try:
    print("Loading trained age/gender model...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load model from {MODEL_PATH}")
    print(f"Error details: {e}")
    print("Ensure the model file exists and training was completed successfully.")
    exit()

if not os.path.exists(HAAR_CASCADE_PATH):
     print(f"FATAL ERROR: Haar Cascade file not found at {HAAR_CASCADE_PATH}")
     print("Download it from the OpenCV repository and place it in the project directory.")
     exit()
try:
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    print("Haar Cascade loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load Haar Cascade classifier: {e}")
    exit()


print("Initializing video capture...")
cap = cv2.VideoCapture(0)
time.sleep(1.0)

if not cap.isOpened():
    print("FATAL ERROR: Could not open video capture device (webcam).")
    exit()

print("Starting real-time detection... Press 'q' to quit.")

if not setup_csv(LOG_FILE):
     print("Warning: Proceeding without CSV logging due to setup error.")

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame from camera. Exiting.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]

        face_batch = preprocess_face(face_roi, IMG_WIDTH, IMG_HEIGHT, CHANNELS)

        if face_batch is None:
            continue

        try:
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

            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)

            label_age = f"Age: {pred_age_range} ({confidence_age:.1f}%)"
            label_gender = f"Gender: {pred_gender} ({confidence_gender:.1f}%)"

            text_y = y - 10 if y - 10 > 10 else y + h + 20

            cv2.putText(frame, label_age, (x, text_y), FONT, FONT_SCALE*0.9, box_color, FONT_THICKNESS)
            cv2.putText(frame, label_gender, (x, text_y + 25), FONT, FONT_SCALE*0.9, box_color, FONT_THICKNESS)

            if not is_allowed:
                cv2.putText(frame, status_message, (x, text_y + 50), FONT, FONT_SCALE, box_color, FONT_THICKNESS+1)


        except Exception as pred_err:
            print(f"Error during prediction or processing face: {pred_err}")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)


    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)


    cv2.imshow('Roller Coaster Security Cam - Press Q to Quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exit key pressed.")
        break

print("Releasing resources...")
cap.release()
cv2.destroyAllWindows()
print("Application finished.")
