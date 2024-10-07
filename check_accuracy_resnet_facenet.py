import cv2
import numpy as np
import time
import pickle
import os
from deepface import DeepFace
from imutils.video import FPS
from sklearn.preprocessing import normalize
from scipy.spatial import distance
import imutils

# Initialize the total images processed and correct predictions counter
total_images = 0
correct_predictions = 0

# Initialize counters
TP = 0  # True Positives
FP = 0  # False Positives
FN = 0  # False Negatives
total_time = 0  # For calculating speed

# Load known encodings and the ResNet face detector
encodingsP = "encodings_facenet.pickle"
print("[INFO] loading encodings + ResNet face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

prototxt = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

# Normalize the known encodings
knownEncodings = normalize(data["encodings"])
knownNames = data["names"]

# Define the dataset path (folder containing subfolders with images to be tested)
dataset_path = "test"  # Change this to the path of your test dataset

# Loop through each subfolder in the dataset folder
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    # Check if the path is a directory
    if not os.path.isdir(person_folder):
        continue

    # Loop through each image in the person's folder
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        print(f"[INFO] Processing image: {image_path}")

        # Load the input image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[WARNING] Skipping {image_path}, cannot read image.")
            continue

        total_images += 1
        frame = imutils.resize(frame, width=500)
        start_time = time.time()  # Start time for speed calculation

        # DNN-based face detection using ResNet model
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        detector.setInput(blob)
        detections = detector.forward()

        # Detect faces
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY, startX:endX]

                try:
                    # Use DeepFace to get face embeddings
                    result = DeepFace.represent(face, model_name='Facenet', enforce_detection=False)
                    
                    for encoding in result:
                        test_embedding = normalize(np.array(encoding['embedding']).reshape(1, -1))[0]
                        best_match_index = -1
                        best_match_distance = float("inf")
                        
                        # Compare with known encodings
                        for index, known_encoding in enumerate(knownEncodings):
                            cosine_distance = distance.cosine(test_embedding, known_encoding)
                            if cosine_distance < best_match_distance:
                                best_match_distance = cosine_distance
                                best_match_index = index

                        recognition_threshold = 0.5  # Adjust based on your testing
                        recognized_name = "Unknown"
                        if best_match_distance < recognition_threshold:
                            recognized_name = knownNames[best_match_index]

                        print(f"[INFO] {image_name} recognized as: {recognized_name}")
                        if recognized_name == person_name:
                            TP += 1  # True Positive
                        elif recognized_name == "Unknown":
                            FN += 1  # False Negative
                        else:
                            FP += 1  # False Positive

                except Exception as e:
                    print(f"[ERROR] {e}. Face detection or embedding failed for {image_path}.")
                    continue

        end_time = time.time()  # End time for speed calculation
        total_time += (end_time - start_time)

# Calculate evaluation metrics
accuracy = (TP / total_images) * 100 if total_images > 0 else 0
precision = (TP / (TP + FP)) if (TP + FP) > 0 else 0
recall = (TP / (TP + FN)) if (TP + FN) > 0 else 0
average_time_per_image = total_time / total_images if total_images > 0 else 0
average_fps = 1 / average_time_per_image if average_time_per_image > 0 else 0

# Output the results
print(f"[INFO] Total images processed: {total_images}")
print(f"[INFO] Accuracy: {accuracy:.2f}%")
print(f"[INFO] Precision: {precision:.2f}")
print(f"[INFO] Recall: {recall:.2f}")
print(f"[INFO] Average FPS: {average_fps:.2f}")
