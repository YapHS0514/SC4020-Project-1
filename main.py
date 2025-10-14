import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

TARGET_IMAGE_FOLDER = 'data/target_images'
DATASET_FOLDER = 'data/images/'

# Load the pre-trained ResNet50 model
# Load ResNet50 with ImageNet weights, exclude the top classification layer (include_top=False)
# and apply global average pooling (pooling='avg').
print("\nLoading ResNet50 model\n")
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Define a function to extract features from an image
def extract_features(img_path):
    """
    Loads and preprocesses an image for ResNet50 and extracts its feature vector.
    """
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to 224x224 pixels
    img_array = image.img_to_array(img)  # Convert image to numpy array
    expanded_img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    preprocessed_img = preprocess_input(expanded_img_array)  # Preprocess for ResNet
    features = model.predict(preprocessed_img)  # Extract features
    return features.flatten()  # Flatten the feature vector for similarity calculation

# Loop through all target images in the target image folder
target_images = [f for f in os.listdir(TARGET_IMAGE_FOLDER) if f.endswith(('.jpg', '.jpeg', '.png'))]

if len(target_images) == 0:
    print("No target images found in the 'target_images' folder.")
    exit()  # Exit if no target images are found

# Process each target image and find the most similar images in the dataset
for target_image in target_images:
    print(f"\nProcessing Target Image: {target_image}\n")
    
    # Define the path to the current target image
    target_image_path = os.path.join(TARGET_IMAGE_FOLDER, target_image)

    # Extract features for the current target image
    print(f"Extracting features from {target_image_path}...\n")
    target_features = extract_features(target_image_path)

    # Extract features for the first 500 images in the dataset folder
    print(f"\nExtracting features from the first 500 images in {DATASET_FOLDER}\n")
    dataset_features = []
    image_filenames = []

    # Loop through all images in the dataset folder and extract their features (limit to 500 images)
    image_count = 0
    for filename in os.listdir(DATASET_FOLDER):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            if image_count < 500:  # Process only the first 500 images
                img_path = os.path.join(DATASET_FOLDER, filename)
                features = extract_features(img_path)
                dataset_features.append(features)
                image_filenames.append(filename)
                image_count += 1
            else:
                break  # Stop once processed 500 images

    # Convert the list of feature vectors to a NumPy array for efficient computation
    dataset_features = np.array(dataset_features)

    # Compute Cosine Similarity 
    print("\nComputing cosine similarity\n")
    # Calculate the cosine similarity between the current target image's features and all the dataset images' features
    similarities = cosine_similarity([target_features], dataset_features)[0]

    # Find and display the most similar images
    # Get the indices of the most similar images, sorted in descending order
    sorted_indices = np.argsort(similarities)[::-1]

    # Display the top 5 most similar images
    print(f"\n--- Top 5 Similar Images for {target_image} ---")
    for i in range(5):
        index = sorted_indices[i]
        filename = image_filenames[index]
        similarity_score = similarities[index]
        print(f"{i+1}. Image: {filename}, Cosine Similarity: {similarity_score:.4f}")
