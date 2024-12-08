# Oshadha

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display image
def display_image(image, title):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load images (assuming they are in the working directory)
healthy_leaf = cv2.imread('healthy_all.jpg')
uploaded_leaf = cv2.imread('upload_all.jpg')

# Resize both images to the same size
size = (400, 600)  # You can adjust this size based on your needs
healthy_leaf_resized = cv2.resize(healthy_leaf, size)
uploaded_leaf_resized = cv2.resize(uploaded_leaf, size)

# Preprocessing: Apply Bilateral Filtering (Noise Reduction) and Sharpening
def preprocess_image(image):
    # Bilateral filter for noise reduction while preserving edges
    bilateral_filtered = cv2.bilateralFilter(image, 9, 75, 75)

    # Sharpening kernel
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])

    # Apply sharpening filter
    sharpened_image = cv2.filter2D(bilateral_filtered, -1, sharpening_kernel)

    return sharpened_image

# Preprocess both healthy and uploaded leaf images
healthy_leaf_preprocessed = preprocess_image(healthy_leaf_resized)
uploaded_leaf_preprocessed = preprocess_image(uploaded_leaf_resized)

# Display preprocessed images
display_image(healthy_leaf_preprocessed, 'Preprocessed Healthy Leaf')
display_image(uploaded_leaf_preprocessed, 'Preprocessed Uploaded Leaf')

# Segmentation: Otsu's Thresholding for Binary Segmentation
def otsu_threshold(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, otsu_thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_thresh

# Apply Otsu's Thresholding
healthy_leaf_thresh = otsu_threshold(healthy_leaf_preprocessed)
uploaded_leaf_thresh = otsu_threshold(uploaded_leaf_preprocessed)

# Display Otsu's Thresholding results
plt.imshow(healthy_leaf_thresh, cmap='gray')
plt.title('Otsu Thresholding - Healthy Leaf')
plt.axis('off')
plt.show()

plt.imshow(uploaded_leaf_thresh, cmap='gray')
plt.title('Otsu Thresholding - Uploaded Leaf')
plt.axis('off')
plt.show()

# Segmentation: Color-based Segmentation using HSV
def color_segmentation(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define a range of green in HSV space (this range can be fine-tuned based on the leaf color)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create a mask for green areas in the image
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Apply the mask to segment the leaf
    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    return segmented_image

# Apply color-based segmentation
healthy_leaf_segmented = color_segmentation(healthy_leaf_preprocessed)
uploaded_leaf_segmented = color_segmentation(uploaded_leaf_preprocessed)

# Display segmented images
display_image(healthy_leaf_segmented, 'Color Segmentation - Healthy Leaf')
display_image(uploaded_leaf_segmented, 'Color Segmentation - Uploaded Leaf')


#Sohan

def highlight_deviations(healthy_preprocessed, unhealthy_preprocessed, deviation_threshold=(30, 40, 40)):
    # Convert both preprocessed images to HSV color space
    healthy_hsv = cv2.cvtColor(healthy_preprocessed, cv2.COLOR_BGR2HSV)
    unhealthy_hsv = cv2.cvtColor(unhealthy_preprocessed, cv2.COLOR_BGR2HSV)

    # Ensure both images are the same size
    if healthy_preprocessed.shape[:2] != unhealthy_preprocessed.shape[:2]:
        unhealthy_hsv = resize_image(unhealthy_hsv, healthy_preprocessed.shape[:2])

    # Detect white areas (holes) in the unhealthy image
    white_mask = cv2.inRange(unhealthy_preprocessed, np.array([240, 240, 240]), np.array([255, 255, 255]))

    # Calculate the absolute difference between the healthy and unhealthy HSV values
    hsv_diff = cv2.absdiff(healthy_hsv, unhealthy_hsv)

    # Apply thresholds to detect significant deviations in hue, saturation, and value
    deviation_mask = (
        (hsv_diff[:, :, 0] > deviation_threshold[0]) |  # Deviation in Hue
        (hsv_diff[:, :, 1] > deviation_threshold[1]) |  # Deviation in Saturation
        (hsv_diff[:, :, 2] > deviation_threshold[2])    # Deviation in Value
    )

    # Exclude white areas (holes) from the deviation mask
    deviation_mask[white_mask > 0] = False

    # Count the number of deviated pixels excluding white areas
    deviation_count = np.sum(deviation_mask)

    # Convert mask to 3-channel format for visualization
    deviation_mask_3channel = np.stack([deviation_mask]*3, axis=-1).astype(np.uint8) * 255

    # Highlight the deviation areas on the unhealthy leaf image (overlay the mask)
    unhealthy_image_highlighted = unhealthy_preprocessed.copy()
    unhealthy_image_highlighted[deviation_mask] = [0, 0, 255]  # Red color for deviations

    # Display the result using matplotlib
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Healthy Leaf (Preprocessed)")
    plt.imshow(cv2.cvtColor(healthy_preprocessed, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Unhealthy Leaf (Preprocessed)")
    plt.imshow(cv2.cvtColor(unhealthy_preprocessed, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Highlighted Deviations")
    plt.imshow(cv2.cvtColor(unhealthy_image_highlighted, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Final decision based on the count of deviations excluding white pixels
    is_deviated = deviation_count > 0
    if is_deviated:
        print(f"Decision: Significant color deviations detected in the unhealthy leaf image ({deviation_count} deviated pixels excluding white areas).")
    else:
        print("Decision: No significant color deviations detected. The unhealthy leaf appears normal.")

    return unhealthy_image_highlighted, is_deviated

# Example usage
highlighted_image, is_deviated = highlight_deviations(healthy_leaf_preprocessed, uploaded_leaf_preprocessed)

# Save the highlighted result
cv2.imwrite('/mnt/data/leaf_deviations_highlighted.jpg', highlighted_image)

# Use the is_deviated value in further processing


# Tharaka

# Step 1: Convert the preprocessed images to grayscale
healthy_leaf_gray = cv2.cvtColor(healthy_leaf_preprocessed, cv2.COLOR_BGR2GRAY)
uploaded_leaf_gray = cv2.cvtColor(uploaded_leaf_preprocessed, cv2.COLOR_BGR2GRAY)

# Step 2: Reuse the Otsu thresholding results
_, healthy_leaf_thresh = cv2.threshold(healthy_leaf_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
_, uploaded_leaf_thresh = cv2.threshold(uploaded_leaf_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 3: Find contours (external contours only)
contours_healthy, _ = cv2.findContours(healthy_leaf_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_uploaded, _ = cv2.findContours(uploaded_leaf_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 4: Draw contours on blank images for both healthy and uploaded leaf
blank_healthy = np.zeros_like(healthy_leaf_gray)
blank_uploaded = np.zeros_like(uploaded_leaf_gray)

# Draw contours (assuming largest contour corresponds to the leaf's outer edge)
cv2.drawContours(blank_healthy, contours_healthy, -1, (255), 2)
cv2.drawContours(blank_uploaded, contours_uploaded, -1, (255), 2)


# Define a boolean variable to store the result
is_pest_infected = False

# Step 5: Use cv2.matchShapes to compare contours
if len(contours_healthy) > 0 and len(contours_uploaded) > 0:
    largest_contour_healthy = max(contours_healthy, key=cv2.contourArea)
    largest_contour_uploaded = max(contours_uploaded, key=cv2.contourArea)

    # cv2.matchShapes returns a similarity score, smaller is better (0 means perfect match)
    similarity = cv2.matchShapes(largest_contour_healthy, largest_contour_uploaded, 1, 0.0)

    # Display similarity score and result
    print(f"Contour Similarity: {similarity}")

    # Define a threshold for significant contour difference
    if similarity > 0.093:
        print("Leaf is infected by pest!")
        is_pest_infected = True
    else:
        print("Leaf is not infected by pest.")
        is_pest_infected = False
else:
    print("No contours found.")

# Step 6: Display images and contours
fig, ax = plt.subplots(2, 2, figsize=(10, 8))

ax[0, 0].imshow(healthy_leaf_gray, cmap='gray')
ax[0, 0].set_title('Healthy Leaf (Grayscale)')

ax[0, 1].imshow(uploaded_leaf_gray, cmap='gray')
ax[0, 1].set_title('Uploaded Leaf (Grayscale)')

ax[1, 0].imshow(blank_healthy, cmap='gray')
ax[1, 0].set_title('Healthy Leaf Outer Edges')

ax[1, 1].imshow(blank_uploaded, cmap='gray')
ax[1, 1].set_title('Uploaded Leaf Outer Edges')

plt.show()


#Avinda

# Function to display image
def display_image(image, title):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Step to Compute the Difference Between Healthy and Uploaded Image
def compute_difference(healthy_thresh, uploaded_thresh):
    # Subtract the uploaded leaf from the healthy leaf to highlight missing regions (holes)
    difference = cv2.absdiff(healthy_thresh, uploaded_thresh)

    # Perform additional thresholding to highlight the differences (holes)
    _, diff_thresh = cv2.threshold(difference, 50, 255, cv2.THRESH_BINARY)

    return diff_thresh

# Compute difference image (highlighting holes)
difference_image = compute_difference(healthy_leaf_thresh, uploaded_leaf_thresh)

# Display the difference image
plt.imshow(difference_image, cmap='gray')
plt.title('Difference Image - Holes Detected')
plt.axis('off')
plt.show()

# Step to Detect and Highlight Holes in the Uploaded Image
def detect_and_highlight_holes(diff_image, original_image):
    # Find contours of the difference image to detect holes
    contours, _ = cv2.findContours(diff_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize is_holes_detected as False
    is_holes_detected = False

    # If any contours are found, set is_holes_detected to True
    if contours:
        is_holes_detected = True

    # Draw contours (holes) on the original uploaded image
    output_image = original_image.copy()
    cv2.drawContours(output_image, contours, -1, (0, 0, 255), 2)  # Red color for holes

    # Print whether holes were detected or not
    print("Holes Detected:", is_holes_detected)

    return output_image, is_holes_detected

# Highlight detected holes on the uploaded image and get the is_holes_detected value
highlighted_image, is_holes_detected= detect_and_highlight_holes(difference_image, uploaded_leaf_resized)

# Display the final result with holes highlighted
display_image(highlighted_image, 'Holes Highlighted in Uploaded Leaf')


#Ravindu

# Function to check if images are loaded
def check_image_loaded(image, image_name):
    if image is None:
        print(f"Error: {image_name} image is not loaded.")
        return False
    return True

# Function to convert images to grayscale
def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function to calculate texture features based on histogram
def calculate_texture_features(image):
    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    hist /= hist.sum()  # Normalize the histogram

    # Calculate features
    contrast = sum((i**2) * hist[i] for i in range(256))
    energy = sum((hist[i] ** 2) for i in range(256))
    homogeneity = sum(hist[i] / (1 + abs(i - j)) for i in range(256) for j in range(256))

    return {
        'contrast': contrast,
        'energy': energy,
        'homogeneity': homogeneity
    }

# Function to display features
def display_features(features, image_label):
    print(f"{image_label} Features:")
    for feature, value in features.items():
        print(f"{feature.capitalize()}: {value:.4f}")
    print()  # Blank line for better readability

# Function to compare features and return a boolean result
def compare_scores(healthy_features, uploaded_features):
    healthy_score = sum(healthy_features.values())
    uploaded_score = sum(uploaded_features.values())

    print("\nComparison Result:")
    if uploaded_score > healthy_score:
        print("The uploaded leaf is likely diseased.")
        print("This indicates that the texture features of the uploaded leaf show higher irregularities compared to the healthy leaf.")
        is_diseased = True
    else:
        print("The uploaded leaf is likely healthy.")
        print("This suggests that the texture features of the uploaded leaf are more consistent with those of a healthy leaf.")
        is_diseased = False

    # Display the individual contributions to the scores
    print("\nFeature Contributions to Scores:")
    print(f"Healthy Leaf Score: {healthy_score:.4f} (Contrast: {healthy_features['contrast']:.4f}, Energy: {healthy_features['energy']:.4f}, Homogeneity: {healthy_features['homogeneity']:.4f})")
    print(f"Uploaded Leaf Score: {uploaded_score:.4f} (Contrast: {uploaded_features['contrast']:.4f}, Energy: {uploaded_features['energy']:.4f}, Homogeneity: {uploaded_features['homogeneity']:.4f})")

    return is_diseased

# Main code execution
if check_image_loaded(healthy_leaf_preprocessed, "Healthy leaf") and check_image_loaded(uploaded_leaf_preprocessed, "Uploaded leaf"):
    healthy_leaf_gray = convert_to_gray(healthy_leaf_preprocessed)
    uploaded_leaf_gray = convert_to_gray(uploaded_leaf_preprocessed)

    # Calculate texture features for both images
    healthy_features = calculate_texture_features(healthy_leaf_gray)
    uploaded_features = calculate_texture_features(uploaded_leaf_gray)

    # Display features
    display_features(healthy_features, "Healthy Leaf")
    display_features(uploaded_features, "Uploaded Leaf")

    # Compare the overall scores and get the boolean result
    is_diseased = compare_scores(healthy_features, uploaded_features)

    # Print the final result based on the boolean value
    print(f"\nIs the uploaded leaf diseased? {'Yes' if is_diseased else 'No'}")

    # Optional: Display the grayscale images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(healthy_leaf_gray, cmap='gray')
    ax[0].set_title('Healthy Leaf (Grayscale)')
    ax[0].axis('off')  # Hide axes

    ax[1].imshow(uploaded_leaf_gray, cmap='gray')
    ax[1].set_title('Uploaded Leaf (Grayscale)')
    ax[1].axis('off')  # Hide axes

    plt.show()
else:
    print("One or both images failed to load. Please check the paths.")


#Avinda

# Rule-based diagnosis logic
def diagnose_leaf(is_deviated, is_pest_infected, is_holes_detected):
    if is_deviated and is_pest_infected and is_holes_detected:
        diagnosis = "Late Blight with Flea Beetles"
        treatment = "Use copper-based fungicides for blight control and insecticidal spray for flea beetles. Remove and destroy severely infected plant parts."
    elif is_deviated and is_pest_infected:
        diagnosis = "Powdery Mildew with Aphid Damage"
        treatment = "Apply a fungicide for powdery mildew and use insecticidal soap for aphid control."
    elif is_deviated and is_holes_detected:
        diagnosis = "Physical Damage or Environmental Stress"
        treatment = "Remove damaged leaves and ensure adequate watering and nutrient supply to support recovery."
    elif is_pest_infected and is_holes_detected:
        diagnosis = "Flea Beetle Infestation"
        treatment = "Apply diatomaceous earth around the plant and use row covers to prevent flea beetle access."
    elif is_deviated:
        diagnosis = "Nutrient Deficiency"
        treatment = "Apply appropriate fertilizer based on soil test results to replenish lacking nutrients."
    elif is_pest_infected:
        diagnosis = "Aphid Infestation"
        treatment = "Spray with insecticidal soap or use a strong water spray to remove aphids."
    elif is_holes_detected:
        diagnosis = "Leaf Miner Damage"
        treatment = "Prune affected leaves and apply neem oil to prevent further spread."
    else:
        diagnosis = "Healthy Plant"
        treatment = "No treatment needed. Continue regular care and monitoring."

    return diagnosis, treatment


# Diagnose the leaf condition and get the treatment recommendation
diagnosis, treatment = diagnose_leaf(is_deviated, is_pest_infected, is_holes_detected)

# Display the results
print("Holes Detected:", is_holes_detected)
print("Deviated:", is_deviated)
print("Pest infected:", is_pest_infected)
print("Diseased:", is_diseased)

print("Diagnosis:", diagnosis)
print("Treatment:", treatment)



