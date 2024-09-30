import os
import shutil
import random

# Define the paths to your dataset and where the split data should be stored
data_dir = 'dataset'  # Path to the original dataset
train_dir = 'train'  # Directory to store training data
test_dir = 'test'  # Directory to store testing data

# Create train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split ratio
split_ratio = 0.7

# Loop through each person folder in the dataset
for person in os.listdir(data_dir):
    person_folder = os.path.join(data_dir, person)

    # Ensure that it's a directory
    if os.path.isdir(person_folder):
        images = os.listdir(person_folder)
        
        # Shuffle the images to ensure random splitting
        random.shuffle(images)
        
        # Calculate the split index
        split_index = int(len(images) * split_ratio)
        
        # Split images into training and testing sets
        train_images = images[:split_index]
        test_images = images[split_index:]
        
        # Create subfolders for the person in the train and test directories
        os.makedirs(os.path.join(train_dir, person), exist_ok=True)
        os.makedirs(os.path.join(test_dir, person), exist_ok=True)
        
        # Move the images to the respective directories
        for image in train_images:
            src = os.path.join(person_folder, image)
            dst = os.path.join(train_dir, person, image)
            shutil.copy2(src, dst)  # Copy the image to the train folder
            
        for image in test_images:
            src = os.path.join(person_folder, image)
            dst = os.path.join(test_dir, person, image)
            shutil.copy2(src, dst)  # Copy the image to the test folder

print("Data split into training and testing sets successfully.")
