from imutils import paths
from deepface import DeepFace
import pickle
import cv2
import os

# our images are located in the dataset folder
print("[INFO] start processing faces...")
imagePaths = list(paths.list_images("train"))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # load the input image
    image = cv2.imread(imagePath)

    # Use DeepFace to analyze the image and get the embedding
    result = DeepFace.represent(image, model_name='Facenet', enforce_detection=False)

    # DeepFace returns a list of encodings, you may want to handle multiple encodings if present
    for encoding in result:
        knownEncodings.append(encoding['embedding'])  # Extract the embedding
        knownNames.append(name)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
with open("encodings_facenet.pickle", "wb") as f:
    pickle.dump(data, f)
