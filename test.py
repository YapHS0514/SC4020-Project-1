import os
import numpy as np
import cv2
from sklearn.metrics import jaccard_score

TARGET_IMAGE_FOLDER = 'data/target_images'  
DATASET_FOLDER = 'data/images/'

# Function to calculate histogram and return it as a normalized 1D array
def compute_histogram(img_path):
    """
    Loads an image, converts to HSV, and computes its normalized histogram.
    Returns the histogram as a flattened array.
    """
    # Load image
    img = cv2.imread(img_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV color space

    # Compute histogram in the HSV space (using 16 bins for each channel)
    hist_hue = cv2.calcHist([img_hsv], [0], None, [16], [0, 256])
    hist_saturation = cv2.calcHist([img_hsv], [1], None, [16], [0, 256])
    hist_value = cv2.calcHist([img_hsv], [2], None, [16], [0, 256])

    # Normalize the histograms
    hist_hue /= hist_hue.sum()
    hist_saturation /= hist_saturation.sum()
    hist_value /= hist_value.sum()

    # Flatten histograms and concatenate them
    hist = np.concatenate([hist_hue.flatten(), hist_saturation.flatten(), hist_value.flatten()])
    return hist

# Preprocess and compute histograms for dataset images
print(f"Extracting histograms from the first 5 images in {DATASET_FOLDER}...\n")
dataset_histograms = []
image_filenames = []

# Loop through the first 5 images in the dataset folder
image_count = 0
for filename in os.listdir(DATASET_FOLDER):
    if filename.endswith(('.jpg', '.jpeg', '.png')) and image_count < 5:
        img_path = os.path.join(DATASET_FOLDER, filename)
        hist = compute_histogram(img_path)
        dataset_histograms.append(hist)
        image_filenames.append(filename)
        image_count += 1

# Convert the list of histograms to a NumPy array for efficient computation
dataset_histograms = np.array(dataset_histograms)

# Preprocess and compute histogram for all target images in the target_image folder
target_images = [f for f in os.listdir(TARGET_IMAGE_FOLDER) if f.endswith(('.jpg', '.jpeg', '.png'))]

if len(target_images) == 0:
    print("No target images found in the 'target_image' folder.")
    exit()  # Exit if no target images are found

# Process all target images and store their feature vectors
target_features_list = []
for target_image in target_images:
    print(f"\nExtracting histogram for target image: {target_image}")
    
    # Define the path to the current target image
    target_image_path = os.path.join(TARGET_IMAGE_FOLDER, target_image)

    # Extract histogram for the current target image
    target_features = compute_histogram(target_image_path)
    target_features_list.append((target_image, target_features))  # Store the features along with the image name

# Calculate Jaccard Similarity for the Target Image vs Dataset Images
for target_image, target_features in target_features_list:
    print(f"\n--- Computing similarities for target image: {target_image} ---")
    
    # Compute Jaccard Similarity 
    print("Computing Jaccard similarity...\n")
    for i, hist in enumerate(dataset_histograms):
        # Calculate Jaccard similarity by comparing histograms as binary sets
        # Threshold histograms to create sets (set non-zero histogram bins to 1)
        target_binary = (target_features > 0).astype(int)
        dataset_binary = (hist > 0).astype(int)
        
        # Calculate the Jaccard similarity (Intersection over Union)
        jaccard_index = np.sum(target_binary & dataset_binary) / np.sum(target_binary | dataset_binary)
        
        print(f"Jaccard Similarity between {target_image} and {image_filenames[i]}: {jaccard_index:.4f}")
