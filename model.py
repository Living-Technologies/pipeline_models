import cv2
import numpy as np
import sys
from PIL import Image

def apply_model(image_path):
    # Open the TIFF file using PIL to handle multi-page TIFFs
    img = Image.open(image_path)
    
    # Initialize an array to store the thresholded images
    thresholded_images = []

    # Loop through each slice (page) in the TIFF
    for i in range(img.n_frames):
        img.seek(i)  # Move to the ith slice
        img_slice = np.array(img)  # Convert the slice to a NumPy array
        
        # Apply Otsu's thresholding
        _, thresholded_slice = cv2.threshold(img_slice, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Append the thresholded slice to the list
        thresholded_images.append(Image.fromarray(thresholded_slice))

    # Save the thresholded images as a new multi-page TIFF
    output_path = image_path.replace('.tif', 'model.tif')
    thresholded_images[0].save(output_path, save_all=True, append_images=thresholded_images[1:])
    print(f"Thresholded image saved as {output_path}")

if __name__ == "__main__":
    image_path = sys.argv[2]  # Get the image path from the command line arguments
    apply_model(image_path)
