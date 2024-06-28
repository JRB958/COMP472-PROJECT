import cv2
import os

# Function to generate synthetic images with multiple transforms
def generate_synthetic_image(original_image_path, output_folder):
    # Load original image
    original_image = cv2.imread(original_image_path)
    
    # Check if the image is loaded correctly
    if original_image is None:
        print(f"Error loading image: {original_image_path}")
        return []

    # Example transforms: Apply multiple operations
    # 1. Horizontal flip
    synthetic_image1 = cv2.flip(original_image, 1)  # Horizontal flip
    
    # 2. Vertical flip
    synthetic_image2 = cv2.flip(original_image, 0)  # Vertical flip
    
    # 3. Gaussian blur
    synthetic_image3 = cv2.GaussianBlur(original_image, (5, 5), 0)  # Gaussian blur with kernel size (5, 5)
    
    # Save synthetic images
    synthetic_image_paths = []

    synthetic_image_filename1 = os.path.basename(original_image_path).replace('.jpg', '_synthetic1.jpg')
    synthetic_image_path1 = os.path.join(output_folder, synthetic_image_filename1)
    cv2.imwrite(synthetic_image_path1, synthetic_image1)
    synthetic_image_paths.append(synthetic_image_path1)
    
    synthetic_image_filename2 = os.path.basename(original_image_path).replace('.jpg', '_synthetic2.jpg')
    synthetic_image_path2 = os.path.join(output_folder, synthetic_image_filename2)
    cv2.imwrite(synthetic_image_path2, synthetic_image2)
    synthetic_image_paths.append(synthetic_image_path2)
    
    synthetic_image_filename3 = os.path.basename(original_image_path).replace('.jpg', '_synthetic3.jpg')
    synthetic_image_path3 = os.path.join(output_folder, synthetic_image_filename3)
    cv2.imwrite(synthetic_image_path3, synthetic_image3)
    synthetic_image_paths.append(synthetic_image_path3)
    
    return synthetic_image_paths

# Function to recursively process all images in the directory structure
def process_directory(directory_path):
    synthetic_images = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                original_image_path = os.path.join(root, file)
                synthetic_image_paths = generate_synthetic_image(original_image_path, root)
                synthetic_images.extend(synthetic_image_paths)
    return synthetic_images

# Example usage
base_directory = 'C:\\Users\\g_alave\\Desktop\\trans_dataset\\categories'  # Replace with your base directory path

# Process all images in the directory structure
synthetic_image_paths = process_directory(base_directory)

# Print paths of saved synthetic images
print("Synthetic images saved:")
for idx, path in enumerate(synthetic_image_paths):
    print(f"{idx + 1}. {path}")
