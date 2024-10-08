import cv2
import time
from imutils.video import FPS
from deepface import DeepFace
import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial import distance
import pickle

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

# Initialize video capture (0 for default camera or provide a video file path)
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize FPS counter
fps = FPS().start()

# Start processing video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Resize the frame for faster processing
    frame = cv2.resize(frame, (500, 500))
    (h, w) = frame.shape[:2]
    
    # DNN-based face detection using ResNet model
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    detector.setInput(blob)
    detections = detector.forward()

    # Detect faces in the frame
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
                    best_match_distance = float("inf")
                    best_match_index = -1

                    # Compare with known encodings
                    for index, known_encoding in enumerate(knownEncodings):
                        cosine_distance = distance.cosine(test_embedding, known_encoding)
                        if cosine_distance < best_match_distance:
                            best_match_distance = cosine_distance
                            best_match_index = index

                    recognition_threshold = 0.5  # Adjust as needed
                    recognized_name = "Unknown"
                    if best_match_distance < recognition_threshold:
                        recognized_name = knownNames[best_match_index]

                    print(f"Recognized as: {recognized_name}")

            except Exception as e:
                print(f"[ERROR] {e}. Face detection or embedding failed.")
    
    # Update the FPS counter
    fps.update()

    # Optional: Show the frame with detections
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the FPS counter and calculate FPS
fps.stop()
print(f"[INFO] Elapsed time: {fps.elapsed():.2f} seconds")
print(f"[INFO] Approx. FPS: {fps.fps():.2f}")

# Release resources
cap.release()
cv2.destroyAllWindows()

