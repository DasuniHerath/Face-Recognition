import cv2
import time
from imutils.video import FPS
from deepface import DeepFace
import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial import distance
import pickle

# Load the known encodings and names from disk
print("[INFO] loading encodings...")
with open("encodings_facenet.pickle", "rb") as f:
    data = pickle.load(f)

# Normalize known encodings
knownEncodings = normalize(data["encodings"])
knownNames = data["names"]

# Initialize Haar Cascade face detector
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture
video_path = "test.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("[ERROR] Could not open video file.")
    exit()

# Initialize the FPS counter
fps = FPS().start()

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] No more frames to read or video ended.")
        break  # Exit loop if no frames are left

    start_time = time.time()  # Start time for speed calculation

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, 
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        try:
            result = DeepFace.represent(face, model_name='Facenet', enforce_detection=True)
            for encoding in result:
                test_embedding = normalize(np.array(encoding['embedding']).reshape(1, -1))[0]
                # Calculate cosine distance with known encodings
                best_match_index = -1
                best_match_distance = float("inf")
                for index, known_encoding in enumerate(knownEncodings):
                    cosine_distance = distance.cosine(test_embedding, known_encoding)
                    if cosine_distance < best_match_distance:
                        best_match_distance = cosine_distance
                        best_match_index = index
                
                # Determine recognition
                recognition_threshold = 0.5  # Adjust based on your testing
                recognized_name = "Unknown"
                if best_match_distance < recognition_threshold:
                    recognized_name = knownNames[best_match_index]
                
                print(f"[INFO] Recognized: {recognized_name}")

            # Draw a bounding box around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box
            # Optionally, put the recognized name above the box
            cv2.putText(frame, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        except Exception as e:
            print(f"[ERROR] {e}. Face detection failed.")

    # Stop the timer
    elapsed_time = time.time() - start_time
    fps.update()  # Update the FPS counter

    # Display the frame with bounding boxes
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit if 'q' is pressed

# Stop the FPS counter
fps.stop()
cap.release()
cv2.destroyAllWindows()

# Print elapsed time and approximate FPS
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

