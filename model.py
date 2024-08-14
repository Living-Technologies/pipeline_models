import cv2
import numpy as np
import sys
from PIL import Image

def apply_model(image_path):
 # Open the TIFF file using PIL to handle multi-page TIFFs
    img = Image.open(image_path)
    
    # Initialize an array to store the edge-detected images
    edge_detected_images = []

    # Loop through each slice (page) in the TIFF
    for i in range(img.n_frames):
        img.seek(i)  # Move to the ith slice
        img_slice = np.array(img)  # Convert the slice to a NumPy array
        
        # Apply Canny edge detection
        edges = cv2.Canny(img_slice, 100, 200)
        
        # Append the edge-detected slice to the list
        edge_detected_images.append(Image.fromarray(edges))

    # Save the edge-detected images as a new multi-page TIFF
    output_path = image_path.replace('.tif', '_canny.tif')
    edge_detected_images[0].save(output_path, save_all=True, append_images=edge_detected_images[1:])
    print(f"Edge-detected image saved as {output_path}")

if __name__ == "__main__":
    image_path = sys.argv[2]  # Get the image path from the command line arguments
    apply_model(image_path)
