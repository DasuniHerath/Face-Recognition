import cv2
import numpy as np

# Load Haar Cascade face detector from OpenCV
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load ResNet SSD face detector model
prototxt = "deploy.prototxt"  # Path to the deploy.prototxt file
model = "res10_300x300_ssd_iter_140000.caffemodel"  # Path to the res10_300x300_ssd_iter_140000.caffemodel file
resnet_detector = cv2.dnn.readNetFromCaffe(prototxt, model)

# Open the video file (replace 'video_file.mp4' with your video file path)
#video_capture = cv2.VideoCapture('test.mp4')
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame from the video
    ret, frame = video_capture.read()
    if not ret:
        break  # Break the loop if the video is done

    # Resize the frame (optional, for better viewing in both windows)
    #frame = cv2.resize(frame, (400, 480))  # Resize for both windows
    frame = cv2.resize(frame, (640, 480))

    # --------- Haar Cascade Detection for the second window (normal feed with Haar Cascade) ----------
    # Convert the frame to grayscale (Haar Cascade works with grayscale images)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    haar_faces = haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Create a copy of the frame for Haar Cascade and draw bounding boxes for detected faces
    haar_detected_frame = frame.copy()
    for (x, y, w, h) in haar_faces:
        cv2.rectangle(haar_detected_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box for Haar detection

    # --------- ResNet Detection for the first window (normal feed with ResNet detection) ----------
    # Prepare the frame for ResNet detection (blob creation)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Perform face detection with ResNet
    resnet_detector.setInput(blob)
    detections = resnet_detector.forward()

    # Create a copy of the frame for ResNet detection and draw bounding boxes for detected faces
    resnet_detected_frame = frame.copy()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Only consider high confidence faces
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(resnet_detected_frame, (startX, startY), (endX, endY), (255, 0, 0), 2)  # Blue box for ResNet detection

    # --------- Show Haar Cascade detection in the second window ----------
    cv2.imshow("Video with Haar Cascade Detection - Window 1", haar_detected_frame)

    # --------- Show ResNet detection in the first window ----------
    cv2.imshow("Video with ResNet Detection - Window 2", resnet_detected_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and close all windows
video_capture.release()
cv2.destroyAllWindows()
