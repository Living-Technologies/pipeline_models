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
        
        # Convert to 8-bit single channel if necessary
        if img_slice.dtype != np.uint8:
            # Normalize the image to 0-255 range, then convert to uint8
            img_slice = cv2.normalize(img_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # If the image has multiple channels (e.g., RGB), convert to grayscale
        if len(img_slice.shape) > 2:
            img_slice = cv2.cvtColor(img_slice, cv2.COLOR_BGR2GRAY)
        
        # Apply Otsu's thresholding
        _, otsu_thresh = cv2.threshold(img_slice, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Append the thresholded slice to the list
        thresholded_images.append(Image.fromarray(otsu_thresh))

    # Save the thresholded images as a new multi-page TIFF
    output_path = image_path.replace('.tif', '_otsu.tif')
    thresholded_images[0].save(output_path, save_all=True, append_images=thresholded_images[1:])
    print(f"Otsu-thresholded image saved as {output_path}")

if __name__ == "__main__":
    image_path = sys.argv[1]  # Get the image path from the command line arguments
    apply_model(image_path)
