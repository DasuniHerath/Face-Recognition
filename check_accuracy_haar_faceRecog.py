# Import necessary packages
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import os
import cv2
import numpy as np
import time

# Initialize the total images processed and correct predictions counter
total_images = 0
correct_predictions = 0

# Initialize counters
TP = 0  # True Positives
FP = 0  # False Positives
FN = 0  # False Negatives
total_time = 0  # For calculating speed

# Load the known faces and embeddings along with OpenCV's Haar cascade for face detection
print("[INFO] loading encodings + face detector...")
encodingsP = "encodings_face_recognition.pickle"
cascade = "haarcascade_frontalface_default.xml"
data = pickle.loads(open(encodingsP, "rb").read())
detector = cv2.CascadeClassifier(cascade)

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

        # Convert the input frame from BGR to grayscale (for face detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the grayscale frame
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
                                          minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

        if len(rects) == 0:
            print(f"[INFO] No faces detected in {image_path}")
            continue
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

        # Compute the facial embeddings for each face bounding box
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        # Loop over the facial embeddings
        for encoding in encodings:
            # Attempt to match each face in the input image to our known encodings
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            distances = face_recognition.face_distance(data["encodings"], encoding)

            # Initialize the name to "Unknown" and set a threshold for confidence
            name = "Unknown"
            threshold = 0.5  # You can adjust this threshold based on your needs

            # Find the best match and its distance
            if True in matches:
                # Get the index of the smallest distance
                best_match_index = np.argmin(distances)
                best_distance = distances[best_match_index]

                # Check if the best distance is below the threshold
                if best_distance < threshold:
                    name = data["names"][best_match_index]

            names.append(name)

        # Select the best name prediction
        predicted_name = "Unknown"  # Default to Unknown
        if names:
            unique_names = list(set(names))  # Get unique names
            best_name = None
            best_score = float('inf')  # Initialize with infinity

            # Loop through unique names to find the best one
            for unique_name in unique_names:
                if unique_name != "Unknown":
                    idx = names.index(unique_name)  # Get index of this name
                    distance = distances[idx]  # Get the corresponding distance
                    if distance < best_score:  # Find the closest match
                        best_score = distance
                        best_name = unique_name

            # Set the predicted name if a best name was found
            if best_name is not None:
                predicted_name = best_name

        # Compare recognized names
        print(f"[INFO] {image_name} recognized as {predicted_name}")

        subfolder_name = person_name  # Get the current person's name from the subfolder

        # Update counters based on the recognition result
        if predicted_name == person_name:
            TP += 1  # True Positive: Correct identification
        elif predicted_name == "Unknown":
            FN += 1  # False Negative: Missed identification
        else:
            FP += 1  # False Positive: Wrong identification

        end_time = time.time()  # End time for speed calculation
        total_time += (end_time - start_time)

# Calculate evaluation metrics
accuracy = (TP / total_images) * 100 if total_images > 0 else 0
precision = (TP / (TP + FP)) if (TP + FP) > 0 else 0
recall = (TP / (TP + FN)) if (TP + FN) > 0 else 0
average_time_per_image = total_time / total_images if total_images > 0 else 0
average_fps = 1 / average_time_per_image if average_time_per_image > 0 else 0

# Final output with accuracy
print(f"[INFO] Total images processed: {total_images}")
print(f"[INFO] Accuracy: {accuracy:.2f}%")
print(f"[INFO] Precision: {precision:.2f}")
print(f"[INFO] Recall: {recall:.2f}")
print(f"[INFO] Average FPS: {average_fps:.2f}")
