from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
import numpy as np

currentname = "unknown"
encodingsP = "encodings.pickle"
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())
prototxt = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

# Replace VideoStream with VideoCapture
video_path = "test.mp4"  # Replace with your video file path
print("[INFO] starting video stream from file...")
vs = cv2.VideoCapture(video_path)

# Check if the video file was successfully opened
if not vs.isOpened():
    print("[ERROR] Could not open video file.")
    exit()

fps = FPS().start()

while True:
    ret, frame = vs.read()

    # Break the loop if no frame is captured (end of the video)
    if not ret:
        break

    frame = imutils.resize(frame, width=500)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    detector.setInput(blob)
    detections = detector.forward()

    boxes = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            boxes.append((startY, endX, endY, startX))

    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        distances = face_recognition.face_distance(data["encodings"], encoding)
        name = "Unknown"
        threshold = 0.5
        if True in matches:
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]

            if best_distance < threshold:
                name = data["names"][best_match_index]

        names.append(name)

    predicted_name = "Unknown"
    if names:
        unique_names = list(set(names))
        best_name = None
        best_score = float('inf')

        for unique_name in unique_names:
            if unique_name != "Unknown":
                idx = names.index(unique_name)
                distance = distances[idx]
                if distance < best_score:
                    best_score = distance
                    best_name = unique_name

        if best_name is not None:
            predicted_name = best_name

    print(f"recognized as {predicted_name}")

    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, predicted_name, (left, y),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)

    cv2.imshow("Facial Recognition is Running", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.release()
