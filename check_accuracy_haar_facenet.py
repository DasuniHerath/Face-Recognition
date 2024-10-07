from imutils import paths
from deepface import DeepFace
import pickle
import cv2
import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial import distance
import os
import imutils
import time

# Initialize the total images processed and correct predictions counter
total_images = 0
correct_predictions = 0

# Initialize counters
TP = 0  # True Positives
FP = 0  # False Positives
FN = 0  # False Negatives
total_time = 0  # For calculating speed

# Load the known encodings and names from disk
print("[INFO] loading encodings...")
with open("encodings_facenet.pickle", "rb") as f:
    data = pickle.load(f)

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
dataset_path = "test"

knownEncodings = normalize(data["encodings"])
knownNames = data["names"]

# Loop over the test images
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        print(f"[INFO] Processing image: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[WARNING] Skipping {image_path}, cannot read image.")
            continue
        total_images += 1
        frame = imutils.resize(frame, width=500)
        start_time = time.time()  # Start time for speed calculation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, 
                                          minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) == 0:
            print(f"[INFO] No faces detected in {image_path}")
            continue
        
        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            try:
                result = DeepFace.represent(face, model_name='Facenet', enforce_detection=True)
                for encoding in result:
                    recognized_name = "Unknown"
                    test_embedding = normalize(np.array(encoding['embedding']).reshape(1, -1))[0]
                    best_match_index = -1
                    best_match_distance = float("inf")
                    for index, known_encoding in enumerate(knownEncodings):
                        cosine_distance = distance.cosine(test_embedding, known_encoding)
                        if cosine_distance < best_match_distance:
                            best_match_distance = cosine_distance
                            best_match_index = index
                    recognition_threshold = 0.5  # Adjust based on your testing
                    if best_match_distance < recognition_threshold:
                        recognized_name = knownNames[best_match_index]
                        print(f"[INFO] Test image {image_path} recognized as: {recognized_name}")
                        subfolder_name = person_name
                        if recognized_name == person_name:
                            TP += 1  # True Positive: Correct identification
                        elif recognized_name == "Unknown":
                            FN += 1  # False Negative: Missed identification
                        else:
                            FP += 1  # False Positive: Wrong identification
                    else:
                        print(f"[INFO] Test image {image_path} is an unknown face.")
                    

            except Exception as e:
                print(f"[ERROR] {e}. Face detection failed for {image_path}.")


# Calculate evaluation metrics
accuracy = (TP / total_images) * 100 if total_images > 0 else 0
precision = (TP / (TP + FP)) if (TP + FP) > 0 else 0
recall = (TP / (TP + FN)) if (TP + FN) > 0 else 0

print(f"[INFO] Total images processed: {total_images}")
print(f"[INFO] Accuracy: {accuracy:.2f}%")
print(f"[INFO] Precision: {precision:.2f}")
print(f"[INFO] Recall: {recall:.2f}")