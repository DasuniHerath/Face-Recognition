from imutils.video import VideoStream
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

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

fps = FPS().start()

while True:
    frame = vs.read()
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

    # Compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        distances = face_recognition.face_distance(data["encodings"], encoding)
        name = "Unknown"
        threshold = 0.5
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
    print(f"recognized as {predicted_name}")

    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom),
            (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, predicted_name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            .8, (0, 255, 255), 2)

    cv2.imshow("Facial Recognition is Running", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
