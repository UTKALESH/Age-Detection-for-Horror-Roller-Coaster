import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import cv2
import pandas as pd
import time

IMG_WIDTH = 64
IMG_HEIGHT = 64
CHANNELS = 3
BATCH_SIZE = 64
EPOCHS = 60 
MODEL_SAVE_PATH = 'saved_models_scratch/age_gender_model.keras'
DATA_DIR = 'data/age_gender' 

AGE_BINS = [0, 13, 20, 30, 40, 50, 61, 120]
NUM_AGE_CLASSES = len(AGE_BINS) - 1
AGE_BIN_LABELS = ['0-12', '13-19', '20-29', '30-39', '40-49', '50-60', '61+']
print(f"Age Bins: {AGE_BIN_LABELS}")
print(f"Number of Age Classes: {NUM_AGE_CLASSES}")

def age_to_bin_index(age):
    for i in range(NUM_AGE_CLASSES):
        if AGE_BINS[i] <= age < AGE_BINS[i+1]:
            return i
    return NUM_AGE_CLASSES - 1

def load_age_gender_dataset(data_dir, img_width, img_height, channels):
    print("--- Loading Age/Gender Dataset ---")
    print("--- CRITICAL: Ensure this function correctly parses YOUR dataset format (filenames/metadata) ---")
    all_images = []
    all_ages = []
    all_genders = []
    image_paths = []
    data_subdirs = ['train', 'test']

    for subdir in data_subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.isdir(subdir_path):
            print(f"Warning: Directory not found: {subdir_path}")
            continue
        print(f"Processing directory: {subdir_path}")
        for filename in os.listdir(subdir_path):
            filepath = os.path.join(subdir_path, filename)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
           
                    parts = filename.split('_')
                    if len(parts) < 2: continue 
                    age = int(parts[0])
                    gender = int(parts[1]) 
                   

                    img = cv2.imread(filepath)
                    if img is None: continue

                    img_resized = cv2.resize(img, (img_width, img_height))
                    if channels == 1:
                         img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                         img_resized = np.expand_dims(img_resized, axis=-1)

                    img_normalized = img_resized.astype('float32') / 255.0

                    all_images.append(img_normalized)
                    all_ages.append(age)
                    all_genders.append(gender)
                    image_paths.append(filepath)

                except ValueError:
                    print(f"Warning: Could not parse age/gender from filename: {filename}")
                except Exception as e:
                    print(f"Error processing file {filepath}: {e}")

    if not all_images:
        raise ValueError("No images loaded! Check dataset path and format.")

    X = np.array(all_images)
    y_age = np.array([age_to_bin_index(age) for age in all_ages])
    y_gender = np.array(all_genders)

    y_age_cat = to_categorical(y_age, num_classes=NUM_AGE_CLASSES)

    print(f"Total images loaded: {len(X)}")

    train_indices = [i for i, path in enumerate(image_paths) if os.path.normpath(path).split(os.sep)[-2] == 'train']
    test_indices = [i for i, path in enumerate(image_paths) if os.path.normpath(path).split(os.sep)[-2] == 'test']

    if not train_indices or not test_indices:
         print("Warning: Could not split based on folder structure. Using random split.")
         X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test = train_test_split(
             X, y_age_cat, y_gender, test_size=0.2, random_state=42, stratify=y_gender
         )
    else:
        print(f"Splitting based on folders: {len(train_indices)} train, {len(test_indices)} test.")
        X_train, y_age_train, y_gender_train = X[train_indices], y_age_cat[train_indices], y_gender[train_indices]
        X_test, y_age_test, y_gender_test = X[test_indices], y_age_cat[test_indices], y_gender[test_indices]

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Training or testing set is empty after splitting.")

    return (X_train, y_age_train, y_gender_train), (X_test, y_age_test, y_gender_test)

def build_model(width, height, channels, num_age_classes):
    input_shape = (height, width, channels)
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), padding="same", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal')(x) 
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    flattened = Flatten()(x)

    common_dense = Dense(256, kernel_initializer='he_normal')(flattened) 
    common_dense = BatchNormalization()(common_dense)
    common_dense = Activation("relu")(common_dense)
    common_dense = Dropout(0.5)(common_dense)

    age_output = Dense(num_age_classes, activation="softmax", name="age_output")(common_dense)
    gender_output = Dense(1, activation="sigmoid", name="gender_output")(common_dense)

    model = Model(inputs=inputs, outputs=[age_output, gender_output], name="age_gender_cnn_scratch")
    return model

if __name__ == "__main__":
    print("--- Training Custom Age/Gender Model From Scratch ---")
    print("Loading and preprocessing data...")
    try:
        (X_train, y_age_train, y_gender_train), (X_test, y_age_test, y_gender_test) = load_age_gender_dataset(
            DATA_DIR, IMG_WIDTH, IMG_HEIGHT, CHANNELS
        )
    except Exception as e:
        print(f"\n--- FATAL ERROR DURING DATA LOADING ---\nError: {e}\n")
        exit()

    print("\nBuilding the model...")
    model = build_model(IMG_WIDTH, IMG_HEIGHT, CHANNELS, NUM_AGE_CLASSES)
    model.summary()

    losses = {"age_output": "categorical_crossentropy", "gender_output": "binary_crossentropy"}
    metrics = {"age_output": "accuracy", "gender_output": "accuracy"}
    loss_weights = {"age_output": 1.0, "gender_output": 1.0}

    print("\nCompiling the model...")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss=losses, loss_weights=loss_weights, metrics=metrics)

    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True) 

    print("\nStarting training...")
    start_time = time.time()
    history = model.fit(
        X_train, {"age_output": y_age_train, "gender_output": y_gender_train},
        batch_size=BATCH_SIZE, validation_split=0.2, epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping], verbose=1
    )
    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f} seconds.")

    print("\nEvaluating on Test Data...")
    try:
        print(f"Loading best model from: {MODEL_SAVE_PATH}")
        best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    except Exception as e:
        print(f"Error loading saved model: {e}. Evaluating with the last epoch model.")
        best_model = model

    test_loss, test_age_loss, test_gender_loss, test_age_acc, test_gender_acc = best_model.evaluate(
        X_test, {"age_output": y_age_test, "gender_output": y_gender_test}, verbose=0
    )
    print(f'\nTest Accuracy (Age Bins): {test_age_acc:.4f}')
    print(f'Test Accuracy (Gender):   {test_gender_acc:.4f}')
    print(f'Overall Test Loss:        {test_loss:.4f}')

    min_accuracy_met = True
    if test_age_acc < 0.70: min_accuracy_met = False
    if test_gender_acc < 0.70: min_accuracy_met = False
    if min_accuracy_met: print("\nModel accuracy meets the minimum 70% requirement for both tasks.")
    else: print("\nWARNING: Model accuracy DOES NOT meet the minimum 70% requirement for one or both tasks!")

    print("\nGenerating Classification Reports and Confusion Matrices...")
    y_pred = best_model.predict(X_test)
    y_pred_age = np.argmax(y_pred[0], axis=1)
    y_pred_gender = (y_pred[1] > 0.5).astype("int32").flatten()
    y_true_age = np.argmax(y_age_test, axis=1)
    y_true_gender = y_gender_test

    print('\n--- Age Classification Report (Bins) ---')
    print(classification_report(y_true_age, y_pred_age, target_names=AGE_BIN_LABELS, zero_division=0))
    print('\n--- Age Confusion Matrix (Bins) ---')
    print(confusion_matrix(y_true_age, y_pred_age))

    print('\n--- Gender Classification Report ---')
    gender_labels = ['Male(0)', 'Female(1)'] 
    print(classification_report(y_true_gender, y_pred_gender, target_names=gender_labels, zero_division=0))
    print('\n--- Gender Confusion Matrix ---')
    print(confusion_matrix(y_true_gender, y_pred_gender))

    print(f"\nModel training complete. Best model saved to {MODEL_SAVE_PATH}")