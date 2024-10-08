from imutils.video import FPS
import face_recognition
import pickle
import cv2
import imutils
import numpy as np
import time

# Load the known faces and embeddings along with OpenCV's Haar cascade for face detection
print("[INFO] loading encodings + face detector...")
encodingsP = "encodings.pickle"  # Path to the encodings file
cascade = "haarcascade_frontalface_default.xml"  # Path to the Haar cascade file
data = pickle.loads(open(encodingsP, "rb").read())  # Load the face encodings from the pickle file
detector = cv2.CascadeClassifier(cascade)  # Load the Haar cascade for face detection

# Initialize counters
total_frames = 0
total_time = 0

# Open the video file or webcam stream
video_path = "test.mp4"  # Change this to your video file path or use 0 for webcam
video_stream = cv2.VideoCapture(video_path)
fps = FPS().start()

# Loop over video frames
while True:
    # Read a frame from the video
    ret, frame = video_stream.read()
    if not ret:
        break  # Exit the loop if no more frames are available

    total_frames += 1
    start_time = time.time()

    # Resize the frame (optional, speeds up processing)
    frame = imutils.resize(frame, width=500)

    # Convert the frame from BGR to grayscale and RGB
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the grayscale frame using Haar cascades
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    # Compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # Loop over the facial embeddings
    for encoding in encodings:
        # Attempt to match each face in the frame to known encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        distances = face_recognition.face_distance(data["encodings"], encoding)

        name = "Unknown"
        threshold = 0.5  # Confidence threshold for recognizing a face

        # Check if we have a match
        if True in matches:
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]

            if best_distance < threshold:
                name = data["names"][best_match_index]

        names.append(name)

    # FPS and timing calculations
    end_time = time.time()
    total_time += (end_time - start_time)

    # Display the frame (optional, for testing purposes)
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update FPS counter
    fps.update()

# Stop the FPS timer and calculate
fps.stop()

# Calculate average FPS and total time
average_fps = fps.fps()
average_time_per_frame = total_time / total_frames if total_frames > 0 else 0

# Release the video stream and close windows
video_stream.release()
cv2.destroyAllWindows()

# Print the FPS and timing results
print(f"[INFO] Total frames processed: {total_frames}")
print(f"[INFO] Average FPS: {average_fps:.2f}")
print(f"[INFO] Average time per frame: {average_time_per_frame:.4f} seconds")

